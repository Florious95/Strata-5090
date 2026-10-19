#include "strata/kernels/verify_kernels.hpp"
#include <algorithm>
#include <cstdio>
#include <vector>

int main() {
    using namespace strata::kernels;
    int checks=0;
    for(int capacity=8;capacity<=16;++capacity)for(int rows=1;rows<=capacity;++rows) {
        std::vector<uint32_t> storage((size_t)capacity*kNonfiniteBlocks+8,0x9ae35712u);
        uint32_t* flags=storage.data()+4;
        for(int row=0;row<rows;++row)for(int b=0;b<kNonfiniteBlocks;++b) {
            std::fill(flags,flags+(size_t)rows*kNonfiniteBlocks,0);
            flags[row*kNonfiniteBlocks+b]=1;
            if(nonfinite_row_mask(flags,rows,capacity)!=(1u<<row))return 1;
            for(int i=0;i<4;++i)if(storage[i]!=0x9ae35712u||storage[storage.size()-1-i]!=0x9ae35712u)return 1;
            ++checks;
        }
        std::fill(flags,flags+(size_t)rows*kNonfiniteBlocks,0);
        if(nonfinite_row_mask(flags,rows,capacity)!=0 || nonfinite_row_mask(flags,capacity+1,capacity)!=UINT32_MAX ||
           nonfinite_row_mask(nullptr,rows,capacity)!=UINT32_MAX || nonfinite_row_mask(flags,0,capacity)!=UINT32_MAX) return 1;
        for(int row=0;row<rows;++row)flags[row*kNonfiniteBlocks]=1;
        if(nonfinite_row_mask(flags,rows,capacity)!=((1u<<rows)-1u))return 1;
        ++checks;
    }
    uint32_t flags[32*kNonfiniteBlocks]={};flags[31*kNonfiniteBlocks+15]=1;
    if(nonfinite_row_mask(flags,32,32)!=0x80000000u||nonfinite_row_mask(flags,1,33)!=UINT32_MAX)return 1;
    std::printf("PASS: %d active-row mask/canary cases, inactive poison ignored; rows9..12 covered, row31 bit safe\n",checks);
}
