// Public CPU ABI/layout contract. Symbolic pointers are never dereferenced.
// This is not CUDA state/attention/streaming parity or an inference benchmark.
#include "strata/core/batch_device_layout.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace strata::core;
using namespace strata::kernels;
namespace {
uint64_t checks = 0;
void check(bool pass, const char* expr, int line) {
    ++checks;
    if (!pass) { std::fprintf(stderr,"batch_device_layout_cpu:%d: %s\n",line,expr); std::abort(); }
}
#define CHECK(x) check(bool(x), #x, __LINE__)
template<class T> T* pointer(uintptr_t value) { return reinterpret_cast<T*>(value); }
constexpr uint64_t sequence=17, table_epoch=9;
constexpr uint32_t profile=0x43444231u;
const std::array<uint64_t,4> generations{40,41,42,43}, owner_epochs{3,4,5,6};

DeviceSlotTable table_fixture() {
    DeviceSlotTable table{};
    for (int b=0;b<4;++b) {
        const uintptr_t base=uintptr_t(0x100000000ull)+uintptr_t(b)*uintptr_t(0x1000000000ull);
        table.target[b].gdn_base=pointer<float>(base);
        table.target[b].ple_hist=pointer<float>(base+uintptr_t(0x100000000ull));
        table.target[b].max_cells=262144;table.target[b].address_epoch=owner_epochs[b];
    }
    for (int qi=0;qi<12;++qi) for (int b=0;b<4;++b) {
        auto& q=table.qsa[qi][b];
        uintptr_t base=uintptr_t(0x10000000000ull)+uintptr_t(qi*4+b)*uintptr_t(0x10000000000ull);
        auto next=[&](){base+=uintptr_t(0x100000000ull);return base;};
        q.pools.k_q4=pointer<uint8_t>(next());q.pools.v_q4=pointer<uint8_t>(next());
        q.map.page_table=pointer<int32_t>(next());q.pools.page_table=q.map.page_table;
        q.host.k_q4=pointer<uint8_t>(next());q.host.v_q4=pointer<uint8_t>(next());
        q.map.slot_block=pointer<int32_t>(next());q.map.slot_stamp=pointer<int32_t>(next());
        q.map.slot_ref=pointer<int32_t>(next());q.map.ctl=pointer<int32_t>(next());
        q.map.miss_block=pointer<int32_t>(next());q.map.miss_slot=pointer<int32_t>(next());
        q.map.n_blocks=65536;q.map.n_slots=5120;
        q.idx_tail=pointer<float>(next());q.idx_dead=pointer<float>(next());q.idx_pooled=pointer<float>(next());
        q.idx_block_pos=pointer<int32_t>(next());q.tail_snapshot=pointer<float>(next());
        q.cos_tab=pointer<float>(uintptr_t(0x500000000000ull));
        q.sin_tab=pointer<float>(uintptr_t(0x500100000000ull)); // readonly sharing is legal
        q.max_cells=262144;q.flags=kDeviceQsaQ4|kDeviceQsaRotated|kDeviceQsaStreamed;q.address_epoch=owner_epochs[b];
    }
    return table;
}
DeviceWindow window(const std::vector<int>& owners,const std::vector<int>& widths,
                    std::array<int,12>& ids,std::array<int64_t,12>& pos) {
    int rows=0;
    for(size_t sg=0;sg<owners.size();++sg)for(int j=0;j<widths[sg];++j) {
        ids[rows]=owners[sg];pos[rows]=257922+owners[sg]*17+j;++rows;
    }
    DeviceWindow out;std::string err;
    const int cut=segment_group_boundary(ids.data(),rows,true,true);
    CHECK(cut>=0);
    CHECK(make_device_window(ids.data(),pos.data(),rows,cut,generations,owner_epochs,
                             sequence,table_epoch,profile,out,err));
    return out;
}
uint64_t recurrence(uint64_t state,int row) { return state*31+uint64_t(row+1); }
}
int main() {
    // Writer capability must come from mutable canonical storage, never from an
    // attention reader by stripping const. Publishing a reader only adds const.
    static_assert(std::is_same_v<decltype(DeviceQsaSlot{}.pools.k_q4), uint8_t*>);
    static_assert(std::is_same_v<decltype(DeviceQsaSlot{}.pools.v_q4), uint8_t*>);
    static_assert(std::is_same_v<decltype(QsaAttnPools{}.k_q4), const uint8_t*>);
    static_assert(!std::is_constructible_v<decltype(DeviceQsaSlot{}.pools), QsaAttnPools>);
    uint8_t owned_k[2]{}, owned_v[2]{}; int32_t owned_map[1]{};
    DeviceQsaSlot writable{};
    writable.pools.k_q4=owned_k;writable.pools.v_q4=owned_v;writable.pools.page_table=owned_map;
    const auto& frozen=writable;
    frozen.pools.k_q4[1]=37;frozen.pools.v_q4[0]=91;
    const auto reader=frozen.pools.read_only();
    CHECK(owned_k[1]==37 && owned_v[0]==91);
    CHECK(reader.k_q4==owned_k && reader.v_q4==owned_v && reader.page_table==owned_map);
    CHECK(!reader.k_pool && !reader.v_pool && !reader.k_q && !reader.v_q && !reader.k_scale && !reader.v_scale);
    static_assert(sizeof(DeviceTargetSlot)==48 && sizeof(DeviceQsaSlot)==288);
    static_assert(sizeof(WindowRow)==16 && sizeof(WindowSegment)==32);
    static_assert(sizeof(DeviceSlotTable)==14016 && sizeof(DeviceWindowLayout)==352 && sizeof(DeviceCommitLayout)==32);
    static_assert(offsetof(WindowSegment,request_generation)==16 && offsetof(WindowSegment,owner_address_epoch)==24);
    static_assert(offsetof(DeviceWindowLayout,sequence)==16 && offsetof(DeviceWindowLayout,table_epoch)==24);
    static_assert(offsetof(DeviceWindowLayout,row)==32 && offsetof(DeviceWindowLayout,segment)==224);
    static_assert(offsetof(DeviceQsaSlot,host)==72 && offsetof(DeviceQsaSlot,map)==136);
    auto table=table_fixture();std::string err;
    CHECK(validate_device_slot_table(table,err));
    DeviceSlotTable mirrored{};std::memcpy(&mirrored,&table,sizeof(table));CHECK(validate_device_slot_table(mirrored,err));
    for(int qi=0;qi<12;++qi)for(int b=0;b<4;++b) {
        CHECK(mirrored.qsa[qi][b].host.k_q4==table.qsa[qi][b].host.k_q4);
        CHECK(mirrored.qsa[qi][b].map.miss_slot==table.qsa[qi][b].map.miss_slot);
        CHECK(mirrored.qsa[qi][b].address_epoch==owner_epochs[b]);
        CHECK(kv_stream_union_fits(mirrored.qsa[qi][b].map.n_slots,8,2051,4));
    }
    auto bad_table=[&](DeviceSlotTable t,const char* label="table mutation"){
        std::string e;const bool accepted=validate_device_slot_table(t,e);
        if(accepted)std::fprintf(stderr,"unexpected accepted table: %s\n",label);
        CHECK(!accepted);CHECK(!e.empty());};
    {auto t=table;t.qsa[0][0].host.k_q4=nullptr;bad_table(t);} {auto t=table;t.qsa[11][3].host.v_q4=nullptr;bad_table(t);}
    {auto t=table;t.qsa[0][0].map.page_table=nullptr;bad_table(t);} {auto t=table;t.qsa[0][0].map.slot_block=nullptr;bad_table(t);}
    {auto t=table;t.qsa[0][0].map.slot_stamp=nullptr;bad_table(t);} {auto t=table;t.qsa[0][0].map.slot_ref=nullptr;bad_table(t);}
    {auto t=table;t.qsa[0][0].map.ctl=nullptr;bad_table(t);} {auto t=table;t.qsa[0][0].map.miss_block=nullptr;bad_table(t);}
    {auto t=table;t.qsa[0][0].map.miss_slot=nullptr;bad_table(t);} {auto t=table;t.qsa[0][0].pools.k_q4=nullptr;bad_table(t);}
    {auto t=table;t.qsa[0][0].map.page_table=pointer<int32_t>(1234);bad_table(t);}
    {auto t=table;t.qsa[0][0].map.n_slots=4000;bad_table(t);} {auto t=table;t.qsa[0][0].map.n_blocks=65535;bad_table(t);}
    {auto t=table;t.qsa[0][0].flags=kDeviceQsaQ4|kDeviceQsaStreamed;bad_table(t);}
    {auto t=table;t.qsa[0][0].tail_snapshot=nullptr;bad_table(t);} {auto t=table;t.qsa[0][0].idx_pooled=nullptr;bad_table(t);}
    {auto t=table;t.qsa[0][0].address_epoch++;bad_table(t);} {auto t=table;t.target[0].address_epoch=0;bad_table(t);}
    {auto t=table;t.target[3].gdn_base=t.target[0].gdn_base;bad_table(t);}
    {auto t=table;t.target[0].gdn_ord0=1;bad_table(t,"nonzero gdn_ord0");}
    {auto t=table;t.target[3].qsa_ord0=1;bad_table(t,"nonzero qsa_ord0");}
    {auto t=table;t.target[0].max_cells=INT64_MAX;
        for(int qi=0;qi<12;++qi){t.qsa[qi][0].max_cells=INT64_MAX;t.qsa[qi][0].map.n_blocks=INT64_MAX/4+1;}
        bad_table(t,"capacity overflow");}
    {auto t=table;t.qsa[0][3].map.ctl=t.qsa[0][0].map.ctl;bad_table(t);}
    {auto t=table;t.qsa[11][0].host.k_q4=t.qsa[0][0].host.k_q4;bad_table(t);}
    // idx_dead is initialized by the pos0 append and is private mutable state.
    {auto t=table;t.qsa[0][3].idx_dead=t.qsa[0][0].idx_dead;bad_table(t,"idx_dead owner alias");}
    uint64_t layouts=0,prefixes=0;
    for(int active=1;active<=4;++active) {
        std::array<int,4> perm{0,1,2,3};
        do {
            if(!std::is_sorted(perm.begin()+active,perm.end()))continue;
            int combinations=1;for(int i=0;i<active;++i)combinations*=8;
            for(int code=0;code<combinations;++code) {
                std::vector<int> owners(perm.begin(),perm.begin()+active),widths;
                int value=code,rows=0;
                for(int i=0;i<active;++i){int width=1+value%8;value/=8;widths.push_back(width);rows+=width;}
                if(rows>12)continue;
                std::array<int,12> ids{};std::array<int64_t,12> pos{};int r=0;
                for(int sg=0;sg<active;++sg)for(int j=0;j<widths[sg];++j){ids[r]=owners[sg];pos[r++]=257922+owners[sg]*17+j;}
                const int cut=segment_group_boundary(ids.data(),rows,true,true);if(cut<0)continue;
                DeviceWindow w;std::string e;
                CHECK(make_device_window(ids.data(),pos.data(),rows,cut,generations,owner_epochs,sequence,table_epoch,profile,w,e));
                CHECK(w.rows==uint32_t(rows)&&w.segments==uint32_t(active)&&w.dense_cut==uint32_t(cut));
                for(int row=0;row<rows;++row)CHECK(w.row[row].slot==ids[row]);
                for(int row=rows;row<12;++row)CHECK(w.row[row].slot==-1);
                for(int sg=active;sg<4;++sg)CHECK(w.segment[sg].slot==-1&&w.segment[sg].length==0);
                if(cut)CHECK(ids[cut-1]!=ids[cut]);
                ++layouts;int combinations_keep=1;for(int width:widths)combinations_keep*=width;
                for(int cp=0;cp<combinations_keep;++cp) {
                    int n=cp,sum=0;std::array<int,4> keeps{};std::array<uint64_t,4> expected{11,22,33,44},actual=expected;
                    for(int sg=0;sg<active;++sg){const auto& s=w.segment[sg];keeps[s.slot]=1+n%s.length;n/=s.length;sum+=keeps[s.slot];
                        for(int j=0;j<keeps[s.slot];++j)expected[s.slot]=recurrence(expected[s.slot],s.first+j);}
                    DeviceCommit c;CHECK(make_device_commit(w,keeps.data(),c,e));
                    CHECK(c.sequence==w.sequence&&c.table_epoch==w.table_epoch);
                    int counted=0;
                    for(int row=0;row<rows;++row)if(w.row[row].local_row<c.keep_by_segment[w.row[row].segment]){
                        actual[w.row[row].slot]=recurrence(actual[w.row[row].slot],row);++counted;}
                    CHECK(counted==sum&&sum>=active&&sum<=rows&&actual==expected);
                    for(int sg=active;sg<4;++sg)CHECK(c.keep_by_segment[sg]==0);
                    ++prefixes;
                }
            }
        }while(std::next_permutation(perm.begin(),perm.end()));
    }
    std::array<int,12> i1{},i2{},i3{};std::array<int64_t,12> p1{},p2{},p3{};
    auto a=window({0,3},{6,6},i1,p1),b=window({3,1,0,2},{3,3,3,3},i2,p2),c=window({2,0,3},{4,4,4},i3,p3);
    CHECK(rows_cut_key(a)==rows_cut_key(b)&&a.segments!=b.segments);
    CHECK(!(rows_cut_key(a)==rows_cut_key(c))&&a.rows==c.rows&&c.dense_cut==8);
    {auto x=a;x.profile++;CHECK(!(rows_cut_key(x)==rows_cut_key(a)));}
    auto bad_window=[&](const int* ids,const int64_t* pos,int rows,int cut,
                         const std::array<uint64_t,4>& g,const std::array<uint64_t,4>& ep,uint64_t seq,uint64_t epoch){
        DeviceWindow out=a;std::string e;CHECK(!make_device_window(ids,pos,rows,cut,g,ep,seq,epoch,profile,out,e));
        CHECK(out.rows==0&&!e.empty());};
    bad_window(i1.data(),p1.data(),0,0,generations,owner_epochs,sequence,table_epoch);
    bad_window(i1.data(),p1.data(),13,6,generations,owner_epochs,sequence,table_epoch);
    bad_window(nullptr,p1.data(),12,6,generations,owner_epochs,sequence,table_epoch);
    bad_window(i1.data(),nullptr,12,6,generations,owner_epochs,sequence,table_epoch);
    bad_window(i1.data(),p1.data(),12,7,generations,owner_epochs,sequence,table_epoch);
    bad_window(i3.data(),p3.data(),12,6,generations,owner_epochs,sequence,table_epoch);
    {std::array<int,12> ids{};std::array<int64_t,12> pos{};
        for(int r=0;r<9;++r)pos[r]=257922+r;
        bad_window(ids.data(),pos.data(),9,0,generations,owner_epochs,sequence,table_epoch);}
    {auto ids=i2;ids[9]=ids[0];bad_window(ids.data(),p2.data(),12,6,generations,owner_epochs,sequence,table_epoch);}
    {auto g=generations;g[0]=0;bad_window(i1.data(),p1.data(),12,6,g,owner_epochs,sequence,table_epoch);}
    {auto ep=owner_epochs;ep[3]=0;bad_window(i1.data(),p1.data(),12,6,generations,ep,sequence,table_epoch);}
    {auto pos=p1;pos[1]+=2;bad_window(i1.data(),pos.data(),12,6,generations,owner_epochs,sequence,table_epoch);}
    {auto pos=p1;pos[0]=-1;bad_window(i1.data(),pos.data(),12,6,generations,owner_epochs,sequence,table_epoch);}
    {auto pos=p1;pos[0]=INT64_MAX;bad_window(i1.data(),pos.data(),12,6,generations,owner_epochs,sequence,table_epoch);}
    {auto ids=i1;ids[0]=4;bad_window(ids.data(),p1.data(),12,6,generations,owner_epochs,sequence,table_epoch);}
    bad_window(i1.data(),p1.data(),12,6,generations,owner_epochs,0,table_epoch);
    bad_window(i1.data(),p1.data(),12,6,generations,owner_epochs,sequence,0);
    auto bad_keep=[&](DeviceWindow w,std::array<int,4> keeps){DeviceCommit out;std::string e;CHECK(!make_device_commit(w,keeps.data(),out,e));CHECK(!e.empty());};
    bad_keep(a,{0,0,0,1});bad_keep(a,{7,0,0,1});bad_keep(a,{1,1,0,1});bad_keep(a,{1,0,-1,1});
    {auto w=a;w.rows=0;bad_keep(w,{1,0,0,1});}
    {DeviceCommit out;std::string e;CHECK(!make_device_commit(a,nullptr,out,e));}
    CHECK(layouts==16736&&prefixes==416676);
    std::printf("batch_device_layout_cpu: %llu checks PASS; layouts=%llu prefixes=%llu; CPU only, no GPU parity\n",
                (unsigned long long)checks,(unsigned long long)layouts,(unsigned long long)prefixes);
}
