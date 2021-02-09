#ifndef __FASTMATH_H__
#define __FASTMATH_H__
#include <cmath>
/*
Utilize quantization and look up table to accelate exp operation
*/
class FastExp{
private:

    static const int MASK_BITS = 12;
    static const int MASK_LEN = (1<<MASK_BITS);
    static const int MASK_VALUE = MASK_LEN - 1;
    static const int QUANT_BITS = 16;
    static const int QUANT_VALUE = (1<<QUANT_BITS);
    static const int QUANT_BOUND = (1 << (2*MASK_BITS - QUANT_BITS)) - 1;

    float neg_coef[2][MASK_LEN];
    float pos_coef[2][MASK_LEN];

public:
    FastExp(){
        for(auto i = 0; i < MASK_LEN; i++){
            neg_coef[0][i] = std::exp(-float(i)/QUANT_VALUE);
            neg_coef[1][i] = std::exp(-float(i)*MASK_LEN/QUANT_VALUE);
            pos_coef[0][i] = std::exp(float(i)/QUANT_VALUE);
            pos_coef[1][i] = std::exp(float(i)*MASK_LEN/QUANT_VALUE);
        }
    }
    ~FastExp(){}

    inline float fexp(const float x){
        int quantX = std::max(std::min(x,float(QUANT_BOUND)),-float(QUANT_BOUND)) * QUANT_VALUE;
        float expx;
        int index;
        if(quantX & 0x80000000){
            index = ~quantX + 0x00000001;
            expx = neg_coef[0][(index)&MASK_VALUE]*neg_coef[1][(index>>MASK_BITS)&MASK_VALUE];
        } else{
            index = quantX;
            expx = pos_coef[0][(index)&MASK_VALUE]*pos_coef[1][(index>>MASK_BITS)&MASK_VALUE];
        }
        return expx;
    }

    inline float sigmoid(float x) {
        return 1.0f / (1.0f + fexp(-x));
    }
};

namespace fastmath{
    static FastExp fastExp;
    inline float exp(const float x){
        return fastExp.fexp(x);
    }

    inline float sigmoid(float x) {
        return fastExp.sigmoid(x);
    }
}

#endif