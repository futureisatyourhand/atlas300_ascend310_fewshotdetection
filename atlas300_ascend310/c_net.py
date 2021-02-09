import tensorflow as tf
import numpy as np

bond_pad_num = 128
BS = 8

def gather(atom, bond_index):
    ### gather (bs, 64, 128) (bs, 2, 128)
    #src, dst = tf.split(bond_index, axis=1, num_or_size_splits=2)
    src = bond_index[:,0]
    dst = bond_index[:,1]
    src_bonds = tf.split(src, axis=0, num_or_size_splits=BS)
    dst_bonds = tf.split(dst, axis=0, num_or_size_splits=BS)
    
    atoms = tf.split(atom, axis=0, num_or_size_splits=BS)
    neighbors = []
    targets = []
    for i in range(BS):
        atm = tf.squeeze(atoms[i])
        neighbors.append( tf.gather(atm, tf.squeeze(dst_bonds[i]), axis=0) )
        targets.append( tf.gather(atm, tf.squeeze(src_bonds[i]), axis=0) )

    neighbors = tf.concat(neighbors, axis=0)
    targets = tf.concat(targets, axis=0)


    
    row_num = []
    for i in range(BS):
        for j in range(src.shape[1]):
             row_num.append([i])
    row_num = tf.constant(row_num)
    scatter = tf.concat((row_num, tf.reshape(src, (-1, 1))), axis=1)
   
    
    return neighbors, targets, src, scatter
        
      

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
 
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def index2seg(index, num=32):
    seg_list = []
    index = tf.expand_dims(index, axis=-1)
    for i in range(num):
        seg_list.append(tf.cast(index==i, dtype=tf.float32))
    return seg_list
    

def scatter_max(bond, index_list, scatter):
    tmp = []
    bond = tf.expand_dims(bond, axis=-1)
    for i in range(len(index_list)):
        seg = index_list[i]*bond
        tmp.append(tf.reduce_max(seg, axis=1))

    cat = tf.concat(tmp, axis=1)
    res = tf.gather_nd(cat, scatter)
    print("scatter:", cat.shape, scatter.shape, res.shape) 
    return tf.reshape(res, (BS, -1)) 


def scatter_sum(bond, index_list, scatter):
    tmp = []
    bond = tf.expand_dims(bond, axis=-1)
    for i in range(len(index_list)):
        seg = index_list[i]*bond
        tmp.append(tf.reduce_sum(seg, axis=1))

    cat = tf.concat(tmp, axis=1)
    res = tf.gather_nd(cat, scatter)
    return tf.reshape(res, (BS, -1)) 

def scatter_sum_only(bond, index_list):
    tmp = []
    bond = tf.expand_dims(bond, axis=-1)
    for i in range(len(index_list)):
        seg = index_list[i]*bond
        tmp.append(tf.reduce_sum(seg, axis=1))

    cat = tf.concat(tmp, axis=1)
    return cat


def matmul128(a, b):
    cols = tf.split(b, axis=-1, num_or_size_splits=128)
    a = tf.reshape(a, (-1, 128, 1))

    rows = []
    # [1024,128,1], [1024,128,1]
    for i in range(bond_pad_num):
        row = tf.reduce_sum(a*cols[i], axis=1)
        rows.append(row)
    res = tf.concat(rows, axis=-1)
    #return tf.reshape(res, (BS, bond_pad_num, 1, 128))
    return res 
       

def inference(atom, bond_index, bond, fingerprint_dim):
    # atom <<<<< torch.Size([64, 128])
    # bond <<<<< torch.Size([128, 10])
    # bond_index <<<<< torch.Size([128, 2])
    batch_size, num_atom , atom_dim = atom.shape
    batch_size, num_bond , bond_dim = bond.shape
    neighbor_atom, target_atom, bond_index0, scatter = gather(atom, bond_index)
    print('atom.shape', atom.shape, bond_index.shape, neighbor_atom.shape, target_atom.shape)

    W_fc1 = weight_variable([10 , fingerprint_dim*fingerprint_dim]) 
    b_fc1 = bias_variable([fingerprint_dim*fingerprint_dim])
    bond = tf.reshape(bond, (-1, 10))
    out_fc1 = tf.matmul(bond, W_fc1) + b_fc1
    #out_fc1 = tf.reshape(-1, 128, fingerprint_dim*fingerprint_dim)
    #out_bn1 = tf.nn.fused_batch_norm(out_fc1,tf.constant(1.0),tf.constant(0.0),epsilon=1e-6)
    bond = tf.reshape(tf.nn.relu(out_fc1), (-1, atom_dim, atom_dim)) 
    print('bond.shape', bond.shape, neighbor_atom.shape, out_fc1.shape)
    ####################################################################################


    #neighbor = tf.reshape(neighbor_atom, (-1, BS, atom_dim))
    #neighbor = tf.matmul(neighbor , bond)
    print(neighbor_atom.shape, bond.shape)
    neighbor = matmul128(neighbor_atom, bond)
    ####################################################################################



    neighbor = tf.reshape(neighbor,(-1, atom_dim))
    target_atom= tf.reshape(target_atom, (-1, atom_dim))
    feature_align = tf.concat([target_atom, neighbor],-1)

    W_fc2 = weight_variable([fingerprint_dim *2 , 1]) 
    b_fc2 = bias_variable([1])
    out_fc2 = tf.matmul(feature_align, W_fc2) + b_fc2
    align_score = tf.nn.leaky_relu(out_fc2)

    W_fc3 = weight_variable([fingerprint_dim,fingerprint_dim]) 
    b_fc3 = bias_variable([fingerprint_dim])
    atend_neigh = tf.matmul(neighbor, W_fc3) + b_fc3
    ####################################################################################
  

 
    align_score = tf.reshape(align_score, (-1, bond_pad_num))
    seg_list = index2seg(bond_index0) 
    ###########scatter##############################
    align_score = align_score - scatter_max(align_score, seg_list, scatter) 
    align_score = tf.exp(align_score)
    ###########scatter##############################
    attention_weight = align_score / scatter_sum(align_score, seg_list, scatter) 

    print("multipy", attention_weight.shape, atend_neigh.shape)
    attention_weight = tf.tile(tf.reshape(attention_weight, (-1, 1)) , (1, atom_dim))
    
    attention = tf.multiply(attention_weight, atend_neigh)
    ###########scatter##############################
    context = scatter_sum_only(attention, seg_list) 
    print(">>>>>>>>over", context.shape)
    return context

   
def run_network():
    #with tf.Graph().as_default():
    with tf.Session(graph=tf.Graph()) as sess:
        atom = tf.placeholder(tf.float32, shape=(BS, 64, 128))
        bond = tf.placeholder(tf.float32, shape=(BS, bond_pad_num, 10))
        bond_index = tf.placeholder(tf.float32, shape=(BS, 2, 10))
        bond_index = tf.Variable(tf.ones([BS,2,bond_pad_num], dtype=tf.int32))
        output = inference(atom, bond_index, bond, 128)
	

        print('FINISH::', output.name) 
        print(sess.graph_def)
        sess.run(tf.global_variables_initializer())
        #constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['add_14'])
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['transpose_1'])
        with tf.gfile.FastGFile('./model.pb', mode='wb') as f:
            f.write(constant_graph.SerializeToString())
        exit()



run_network()


