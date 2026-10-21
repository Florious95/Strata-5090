#include "strata/core/dspark_confidence.hpp"
#include "strata/core/batch_segments.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>

using strata::core::DsparkConfidence;
using strata::core::DsparkDepthPlan;
static size_t checks = 0;
#define CHECK(x) do { ++checks; if (!(x)) { std::fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#x); std::exit(1); } } while (0)
static_assert(DsparkConfidence::max_rows == strata::kernels::kVerifyMaxT);
static_assert(std::is_trivially_copyable_v<DsparkConfidence>);
static_assert(sizeof(DsparkConfidence) < 512);

struct Observation { uint64_t window; int useful, kept; };
DsparkDepthPlan reference(const std::vector<Observation>& log, uint64_t generation, int maximum) {
    DsparkDepthPlan p;
    p.generation = generation; p.step = log.size()+1; p.rows = std::min(3,maximum);
    if (log.size() < 2) return p;
    const size_t end = log.size()-1, begin = end > 16 ? end-16 : 0;
    std::array<int,8> trials{}, success{};
    int total_trials=0,total_success=0;
    for (size_t i=begin;i<end;++i) {
        ++p.observations; p.history_through_window=log[i].window;
        for (int d=1;d<log[i].useful;++d) {
            if (log[i].kept<d) break;
            ++trials[d]; ++total_trials;
            success[d] += log[i].kept>d; total_success += log[i].kept>d;
        }
    }
    if (total_trials) p.observed_rate=double(total_success)/total_trials;
    if (p.observations<4 || !total_trials) return p;
    const double unseen=double(total_success+1)/(total_trials+2);
    double prefix=1;
    p.rows=std::min(2,maximum);
    for (int d=1;d<maximum;++d) {
        prefix *= trials[d] ? double(success[d]+1)/(trials[d]+2) : unseen;
        if (prefix<0.5) break;
        p.rows=std::max(p.rows,d+1);
    }
    return p;
}
void same(const DsparkDepthPlan& a,const DsparkDepthPlan& b) {
    CHECK(a.rows==b.rows); CHECK(a.observations==b.observations);
    CHECK(a.generation==b.generation); CHECK(a.step==b.step);
    CHECK(a.history_through_window==b.history_through_window);
    CHECK(a.observed_rate==b.observed_rate);
}
uint32_t random_u32(uint32_t& x) { x^=x<<13;x^=x>>17;x^=x<<5;return x; }

int main() {
    DsparkConfidence empty;
    CHECK(empty.plan(8).rows==0); CHECK(!empty.reset(0));
    CHECK(empty.reset(7));
    for (int maximum=1;maximum<=8;++maximum) {
        const auto p=empty.plan(maximum);
        CHECK(p.rows==std::min(3,maximum)); CHECK(p.step==1);
        CHECK(p.observations==0 && p.history_through_window==0 && p.generation==7);
    }
    CHECK(empty.plan(0).rows==0 && empty.plan(9).rows==0);
    const auto initial=empty.plan(8);
    CHECK(!empty.observe(0,1,3,3)); CHECK(!empty.observe(8,1,3,3));
    CHECK(!empty.observe(7,0,3,3)); CHECK(!empty.observe(7,1,0,0));
    CHECK(!empty.observe(7,1,9,1)); CHECK(!empty.observe(7,1,3,0));
    CHECK(!empty.observe(7,1,3,4)); same(initial,empty.plan(8));
    CHECK(empty.observe(7,2,8,8));
    const auto once=empty.plan(8);
    CHECK(once.rows==3 && once.observations==0 && once.history_through_window==0 && once.step==2);
    CHECK(!empty.observe(7,2,8,8)); CHECK(!empty.observe(7,1,8,8));
    same(once,empty.plan(8));
    CHECK(empty.observe(7,8,8,8));
    CHECK(empty.plan(8).observations==1 && empty.plan(8).history_through_window==2);

    // The immediately preceding outcome may change arbitrarily, including
    // offered length, without changing the next step's confidence decision.
    DsparkConfidence prefix;
    CHECK(prefix.reset(11));
    for (uint64_t n=1;n<=24;++n) CHECK(prefix.observe(11,n*5,8,n%2 ? 8:1));
    for (int useful=1;useful<=8;++useful) for (int kept=1;kept<=useful;++kept) {
        auto left=prefix,right=prefix;
        CHECK(left.observe(11,200,8,8)); CHECK(right.observe(11,200,useful,kept));
        for (int maximum=1;maximum<=8;++maximum) same(left.plan(maximum),right.plan(maximum));
    }
    DsparkConfidence high,low;
    CHECK(high.reset(1) && low.reset(1));
    for (uint64_t n=1;n<=18;++n) { CHECK(high.observe(1,n,8,8));CHECK(low.observe(1,n,8,1)); }
    CHECK(high.plan(8).rows==8 && high.plan(8).observed_rate==1);
    CHECK(low.plan(8).rows==2 && low.plan(8).observed_rate==0);
    for (int cap=1;cap<=8;++cap) CHECK(high.plan(cap).rows==cap);
    CHECK(high.reset(2)); CHECK(high.plan(8).rows==3 && high.plan(8).step==1);
    CHECK(!high.observe(1,99,8,8));

    // Reference uses an unbounded vector, not the production ring. Exercise
    // wrap, censorship, sparse global window IDs and a changing maximum.
    uint32_t rng=0x5090c4;
    for (uint64_t generation=1;generation<=40;++generation) {
        DsparkConfidence h; CHECK(h.reset(generation));
        std::vector<Observation> log; uint64_t window=0;
        for (int n=0;n<1000;++n) {
            const int maximum=1+random_u32(rng)%8;
            same(h.plan(maximum),reference(log,generation,maximum));
            const int useful=1+random_u32(rng)%8,kept=1+random_u32(rng)%useful;
            window+=1+random_u32(rng)%8;
            CHECK(h.observe(generation,window,useful,kept));log.push_back({window,useful,kept});
            const auto p=h.plan(8);
            CHECK(p.rows>=2 && p.rows<=8);CHECK(p.observations<=16);
            CHECK(p.history_through_window<window);
        }
    }
    // No owner can lend its acceptance history to another request/slot.
    std::array<DsparkConfidence,4> owners;
    for (int b=0;b<4;++b) CHECK(owners[b].reset(100+b));
    for (uint64_t w=1;w<=24;++w) {
        const auto untouched=owners[3].plan(8);
        CHECK(owners[0].observe(100,w,8,8)); CHECK(owners[1].observe(101,w,8,1));
        same(untouched,owners[3].plan(8));
    }
    CHECK(owners[0].plan(8).rows==8);CHECK(owners[1].plan(8).rows==2);
    same(owners[2].plan(8),reference({},102,8));
    CHECK(owners[0].reset(900));CHECK(owners[0].plan(8).rows==3);
    CHECK(!owners[0].observe(100,100,8,8));CHECK(owners[1].plan(8).rows==2);

    // A high-confidence chain must actually reach S8 under its own decisions,
    // not only when an artificial producer always offers eight rows.
    DsparkConfidence online; CHECK(online.reset(1)); int largest=0;
    for (uint64_t n=1;n<=1000;++n) {
        const int rows=online.plan(8).rows;largest=std::max(largest,rows);
        CHECK(online.observe(1,n,rows,rows));
    }
    CHECK(largest==8);
    for (uint64_t n=1001;n<=1036;++n) CHECK(online.observe(1,n,8,1));
    CHECK(online.plan(8).rows==2);
    for (uint64_t n=1037;n<=1072;++n) CHECK(online.observe(1,n,8,8));
    CHECK(online.plan(8).rows==8);

    using namespace strata::core;
    for (int active=1;active<=4;++active) {
        int wanted[4]={},allocated[4]={};
        for(int b=0;b<active;++b) wanted[b]=8;
        allocate_batch_rows(wanted,4,12,0,allocated);
        CHECK(pack_segment_rows(allocated,4,0)>=0);
        for(int b=0;b<active;++b) CHECK(allocated[b]==(active==1?8:12/active));
    }
    for(int code=0;code<6561;++code) {
        int q=code,wanted[4]={};
        for(int b=0;b<4;++b) { wanted[b]=q%9;q/=9; }
        int active=0;for(int x:wanted)active+=x>0;
        if(!active)continue;
        for(int budget=active;budget<=12;++budget) for(int first=0;first<4;++first) {
            int allocated[4]={},ids[12]={};int64_t positions[12]={};
            allocate_batch_rows(wanted,4,budget,first,allocated);
            const int total=pack_segment_rows(allocated,4,first);CHECK(total>=active && total<=budget);
            int n=0;
            for(int b=0;b<4;++b) {
                CHECK(allocated[b]>=int(wanted[b]>0) && allocated[b]<=wanted[b] && allocated[b]<=8);
                for(int j=0;j<allocated[b];++j) { ids[n]=b;positions[n++]=258000+j; }
            }
            CHECK(n==total && segment_group_boundary(ids,n)>=0);
            std::string error;CHECK(validate_segments(ids,positions,n,12,4,true,error));
        }
    }
    std::printf("dspark_confidence_cpu: %zu checks passed; N-2 lag, 16+1 history, generation isolation, S8 and rows12 packing\n",checks);
}
