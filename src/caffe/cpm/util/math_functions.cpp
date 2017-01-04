#include "caffe/cpm/util/math_functions.hpp"

namespace caffe {

int updiv(const int a, const int b){
    return (a+b-1)/b;
}

}  // namespace caffe
