#include "strata/core/total_q_buckets.hpp"
#include "strata/core/batch_device_layout.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <vector>

namespace c = strata::core;
static size_t checks = 0, plans = 0, shortened = 0;
static void check(bool ok, const char* message) {
    ++checks;
    if (!ok) { std::fprintf(stderr,"FAIL total-Q: %s\n",message); std::exit(1); }
}
static int cut_of(const std::array<int,4>& w) {
    const int n=w[0]+w[1]+w[2]+w[3];
    int sum=0, answer=0, best=100;
    for (int b=0;b<4;++b) {
        if (!w[b]) continue;
        sum+=w[b]; if (sum==n) break;
        if (n<=8) {
            const int loss=std::abs(2*sum-n);
            if (loss<best) { best=loss; answer=sum; }
        } else if (sum<=8 && n-sum<=8) answer=sum;
    }
    return answer;
}
static bool preferred(const std::array<int,4>& a,const std::array<int,4>& b,int first) {
    for (int j=0;j<4;++j) {
        const int owner=(first+j)%4;
        if (a[owner]!=b[owner]) return a[owner]>b[owner];
    }
    return false;
}
int main() {
    std::set<std::array<int,4>> keys;
    for (size_t i=0;i<c::kTotalQTemplates.size();++i) {
        const auto& t=c::kTotalQTemplates[i];
        check(t.rows<=12 && t.cut>=1 && t.cut<=8 && t.rows-t.cut<=8,"G2 dense bounds");
        check(c::total_q_template_id(t.rows,t.cut)==(int)i,"template identity");
        check(cut_of(t.witness)==t.cut,"production witness cut");
        keys.insert(c::total_q_key_prefix(t));
    }
    check(keys.size()==8,"actual rows must not collide within one bucket/cut");
    std::set<int> families;
    for (const auto& t:c::kTotalQTemplates) families.insert(t.bucket);
    check(families==std::set<int>({3,6,9,12}),"four bounded families");
    check(c::total_q_template_id(12,5)<0 && c::total_q_template_id(8,4)<0,"reject undeclared shapes");
    // Exercise the same key classifier/count helper used by production LRU
    // and OOM eviction: admission churn must never evict a warmed template.
    std::map<std::vector<int>,int> bank;
    for (int id=0;id<8;++id) {
        const auto p=c::total_q_key_prefix(c::kTotalQTemplates[id]);
        const std::vector<int> key(p.begin(),p.end());
        check(c::total_q_pinned_key(key),"production pin classification");bank[key]=id+1;
    }
    check(!c::total_q_pinned_key({c::kTotalQTemplateKeyTag}) &&
          !c::total_q_pinned_key({c::kTotalQTemplateKeyTag,9,2,1}),"malformed keys are not pinned");
    for (int tick=0;tick<1000;++tick) {
        if (c::total_q_evictable_count(bank)>=8) {
            auto victim=bank.end();
            for (auto it=bank.begin();it!=bank.end();++it)
                if (!c::total_q_pinned_key(it->first) && (victim==bank.end() || it->second<victim->second)) victim=it;
            check(victim!=bank.end(),"evictable positive control");bank.erase(victim);
        }
        bank[{tick,1,0}]=100+tick;
        for (const auto& p:keys) check(bank.count(std::vector<int>(p.begin(),p.end()))==1,"all dynamic replay keys remain hot");
        check(c::total_q_evictable_count(bank)<=8 && bank.size()<=16,"separate bounded legacy LRU");
    }
    for (auto it=bank.begin();it!=bank.end();) {
        if (!c::total_q_pinned_key(it->first)) it=bank.erase(it);else ++it;
    }
    check(bank.size()==8 && c::total_q_evictable_count(bank)==0,"OOM cannot silently evict sealed templates");
    std::array<size_t,8> seen{};
    for (int code=0;code<6561;++code) {
        int x=code,total=0,active=0; std::array<int,4> wanted{};
        for (int& n:wanted) { n=x%9; x/=9; total+=n; active+=n>0; }
        for (int first=0;first<4;++first) {
            c::TotalQPlan p; std::string error;
            const bool ok=c::plan_total_q_buckets(wanted,first,p,error);
            if (active<2 || total>12) {
                check(!ok && !error.empty() && p.template_id==-1,"invalid geometry fails closed");
                continue;
            }
            check(ok && error.empty(),"every legal multi-owner demand has a pending-only fallback");
            const auto& t=c::kTotalQTemplates[p.template_id];
            int rows=0;
            for (int b=0;b<4;++b) {
                check(p.widths[b]>=int(wanted[b]>0) && p.widths[b]<=wanted[b],"prefix-only, inactive-zero, no starvation");
                rows+=p.widths[b];
            }
            check(rows==t.rows && cut_of(p.widths)==t.cut,"chosen exact template");
            check(rows<=total && rows<=12,"physical twelve-row capacity");
            shortened+=rows<total; ++plans; ++seen[p.template_id];
            // Independent finite oracle: reject any higher-row or better tied
            // vector, using an independently calculated dense boundary.
            for (int a=wanted[0]>0;a<=wanted[0];++a)
            for (int b=wanted[1]>0;b<=wanted[1];++b)
            for (int d=wanted[2]>0;d<=wanted[2];++d)
            for (int e=wanted[3]>0;e<=wanted[3];++e) {
                const std::array<int,4> q{a,b,d,e}; const int n=a+b+d+e;
                if (c::total_q_template_id(n,cut_of(q))<0) continue;
                check(n<rows || (n==rows && !preferred(q,p.widths,first)),"max retained work and rotating tie policy");
            }
            std::array<int,4> owners{0,1,2,3};
            do {
                int ids[12]={}, keep[4]={}; int64_t positions[12]={}; int at=0;
                for (int b=0;b<4;++b) {
                    const int owner=owners[b]; keep[owner]=p.widths[b] ? 1+(first+b)%p.widths[b] : 0;
                    for (int j=0;j<p.widths[b];++j) { ids[at]=owner; positions[at++]=257900+j; }
                }
                c::DeviceWindow w; c::DeviceCommit commit;
                check(c::make_device_window(ids,positions,rows,t.cut,{11,12,13,14},{21,22,23,24},plans,1,286,w,error),"production Window ABI accepts every owner permutation");
                check(c::make_device_commit(w,keep,commit,error),"production accepted-prefix commit remains valid");
                check(c::rows_cut_key(w).rows==(unsigned)t.rows && c::rows_cut_key(w).cut==(unsigned)t.cut,"owner/nseg not key data");
                int sum=0; for (unsigned s=0;s<w.segments;++s) sum+=commit.keep_by_segment[s];
                check(sum>=active && sum<=rows,"accepted keep conservation");
            } while (std::next_permutation(owners.begin(),owners.end()));
        }
    }
    for (size_t n:seen) check(n>0,"all eight templates have positive controls");
    for (auto wanted : {std::array<int,4>{6,6,0,0},{3,3,3,3},{4,4,4,0}}) {
        c::TotalQPlan p; std::string error;
        check(c::plan_total_q_buckets(wanted,0,p,error) && p.widths==wanted,"golden full-load work unchanged");
    }
    { c::TotalQPlan p; p.widths={3,3,3,3};std::string e;
      check(c::plan_total_q_buckets(p.widths,0,p,e) && p.widths==std::array<int,4>{3,3,3,3},"input/output alias");
      check(!c::plan_total_q_buckets({9,1,0,0},0,p,e),"reject above kVerifyMaxT");
      check(!c::plan_total_q_buckets({1,1,0,0},4,p,e),"reject invalid rotation"); }
    std::printf("PASS total-Q: %zu checks, %zu dynamic plans, %zu prefix-shortened plans; four families/eight exact G2 templates; production ABI/commit/owner permutations. CPU only.\n",checks,plans,shortened);
}
