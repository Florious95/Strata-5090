// Synthetic full-Q4 Host/map/indexer/selection/attention contract, no weights.
// CLOCK placement is not compared bitwise: concurrent miss-list order is not an
// arithmetic contract. Each selected logical page must equal authoritative Host.
#include "strata/core/batch_device_layout.hpp"
#include "strata/kernels/batch_device_ops.hpp"
#include "strata/kernels/idx_pooled.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/qsa.hpp"
#include <cuda_runtime.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <vector>
namespace k = strata::kernels;
namespace c = strata::core;
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(2); }
}
void require(bool ok, const char* what) { if (!ok) { std::fprintf(stderr,"FAIL %s\n",what);std::exit(1); } }
template<class T> struct Buffer {
    T* p = nullptr; size_t size;
    explicit Buffer(size_t n) : size(n) { ck(cudaMalloc((void**)&p,n*sizeof(T)),"alloc");zero(); }
    ~Buffer() { cudaFree(p); }
    Buffer(const Buffer&) = delete;
    void zero() { ck(cudaMemset(p,0,size*sizeof(T)),"zero"); }
    void upload(const std::vector<T>& v) { require(v.size()==size,"upload shape");ck(cudaMemcpy(p,v.data(),size*sizeof(T),cudaMemcpyHostToDevice),"upload"); }
    void object(const T& value) { require(size==1,"object shape");ck(cudaMemcpy(p,&value,sizeof(T),cudaMemcpyHostToDevice),"object"); }
    std::vector<T> download() const { std::vector<T> v(size);ck(cudaMemcpy(v.data(),p,size*sizeof(T),cudaMemcpyDeviceToHost),"download");return v; }
};
struct Host {
    uint8_t *p=nullptr,*device=nullptr;size_t size;std::vector<uint8_t> initial;
    Host(size_t n,int owner,int plane):size(n) {
        ck(cudaHostAlloc((void**)&p,n,cudaHostAllocMapped),"Host alloc");
        ck(cudaHostGetDevicePointer((void**)&device,p,0),"Host map");
        for(size_t i=0;i<n;i+=18) { p[i]=0;p[i+1]=0x28;for(int j=2;j<18;++j)p[i+j]=uint8_t(0x78+owner*17+plane*5+(i/18)%13); }
        initial.assign(p,p+n);
    }
    void reset(){std::memcpy(p,initial.data(),size);}
    ~Host(){cudaFreeHost(p);} Host(const Host&)=delete;
};
float rf(uint32_t& s){s^=s<<13;s^=s>>17;s^=s<<5;return float(int32_t(s))/2147483648.0f;}
constexpr int Cells=262144, Slots=5120, Blocks=Cells/4, Rows=k::kDeviceBatchRows, PBytes=4*2*144;
const k::QsaShapes Shape=k::qsa_real_shapes();
const int Cap=(int)k::qsa_selection_width(k::kTopkMaxCells,Shape), MaxBlocks=Blocks+2;
const size_t Stride=k::qsa_decode_attn_scratch_floats(Cap,Shape);
struct Owner {
    Host hk,hv;
    Buffer<uint8_t> kp{size_t(Slots)*PBytes},vp{size_t(Slots)*PBytes};
    Buffer<int32_t> page{Blocks},block{Slots},stamp{Slots},ref{Slots},ctl{k::kKvCtlInts},miss{Slots},where{Slots};
    Buffer<float> tail{384},dead{128},pooled{size_t(MaxBlocks)*128},snap{384};Buffer<int32_t> idxpos{1};
    k::DeviceQsaSlot q{};std::vector<float> initial_pooled,initial_dead;
    Owner(int b):hk(size_t(Cells)*2*144,b,0),hv(size_t(Cells)*2*144,b,1) {
        q.pools.k_q4=kp.p;q.pools.v_q4=vp.p;q.pools.page_table=page.p;
        q.host.k_q4=hk.device;q.host.v_q4=hv.device;
        q.map={page.p,block.p,stamp.p,ref.p,ctl.p,miss.p,where.p,Blocks,Slots};
        q.idx_tail=tail.p;q.idx_dead=dead.p;q.idx_pooled=pooled.p;q.idx_block_pos=idxpos.p;
        q.tail_snapshot=snap.p;q.max_cells=Cells;q.address_epoch=1;q.flags=7;
        uint32_t seed=0x27193+b*1337;
        initial_pooled.resize(pooled.size);for(auto& f:initial_pooled)f=rf(seed);
        if(k::qsa_idx_pooled_bf16()) {   // C8-6: a BF16 store holds BF16 keys (the buffer's first half)
            std::vector<uint16_t> h(pooled.size);for(auto& x:h)x=k::bf16_from_f32(rf(seed));
            std::memcpy(initial_pooled.data(),h.data(),h.size()*2);
        }
        pooled.upload(initial_pooled);
        initial_dead.resize(128);for(auto& f:initial_dead)f=rf(seed);dead.upload(initial_dead);
    }
    void reset(cudaStream_t stream) {
        hk.reset();hv.reset();pooled.upload(initial_pooled);dead.upload(initial_dead);snap.zero();
        k::kv_stream_reset(q.map,stream);
        ck(cudaMemsetAsync(kp.p,0,kp.size,stream),"reset K");
        ck(cudaMemsetAsync(vp.p,0,vp.size,stream),"reset V");
        ck(cudaMemsetAsync(tail.p,0,tail.size*4,stream),"reset tail");
        ck(cudaMemsetAsync(idxpos.p,0,4,stream),"reset block pos");
    }
    void check_map() {
        const auto pt=page.download(),sb=block.download();
        for(int b=0;b<Blocks;++b) require(pt[b]>=-1 && pt[b]<Slots &&
            (pt[b]<0 || sb[pt[b]]==b),"complete page map bijection, no pending claim");
        for(int s=0;s<Slots;++s) require(sb[s]>=-1 && sb[s]<Blocks &&
            (sb[s]<0 || pt[sb[s]]==s),"complete slot map inverse");
    }
    void check_appended_pages(const std::vector<int32_t>& steps,const k::WindowSegment& s) {
        const auto pt=page.download();const auto keys=kp.download(),values=vp.download();
        for(int r=s.first;r<s.first+s.length;++r) {
            const int b=steps[r*k::kStepCount+k::kStepPos]/4,slot=pt[b];
            require(slot>=0&&slot<Slots,"append fixture page was already resident");
            require(!std::memcmp(keys.data()+size_t(slot)*PBytes,hk.p+size_t(b)*PBytes,PBytes),"resident append K mirrored into pool");
            require(!std::memcmp(values.data()+size_t(slot)*PBytes,hv.p+size_t(b)*PBytes,PBytes),"resident append V mirrored into pool");
        }
    }
    void check_pages(const std::vector<int32_t>& ids,const std::vector<int32_t>& steps,const k::WindowSegment& s) {
        const auto pt=page.download(),sb=block.download();const auto keys=kp.download(),values=vp.download();
        for(int r=s.first;r<s.first+s.length;++r)for(int j=0;j<steps[r*k::kStepCount+k::kStepWidth];++j) {
            const int b=ids[size_t(r)*Cap+j]/4,slot=pt[b];
            require(slot>=0&&slot<Slots&&sb[slot]==b,"selected page mapping bijection");
            require(!std::memcmp(keys.data()+size_t(slot)*PBytes,hk.p+size_t(b)*PBytes,PBytes),"resident K equals logical Host page");
            require(!std::memcmp(values.data()+size_t(slot)*PBytes,hv.p+size_t(b)*PBytes,PBytes),"resident V equals logical Host page");
        }
    }
};
struct Work {
    Buffer<float> raw{Rows*128},queries{Rows*512},query{Rows*24*256},key{Rows*2*256},value{Rows*2*256};
    Buffer<float> scores{size_t(Rows)*MaxBlocks},scratch{Rows*Stride},out{Rows*24*256};
    Buffer<int32_t> steps{Rows*k::kStepCount},ids{size_t(Rows)*Cap};
    void inputs(uint32_t seed,const std::vector<int32_t>& st) {
        auto fill=[&](Buffer<float>& b,float scale){std::vector<float>x(b.size);for(auto&f:x)f=scale*rf(seed);b.upload(x);};
        fill(raw,0.2f);fill(queries,0.1f);fill(query,0.1f);fill(key,0.2f);fill(value,0.2f);
        steps.upload(st);ck(cudaMemset(scores.p,0x7f,scores.size*4),"score poison");
        ck(cudaMemset(ids.p,0x7f,ids.size*4),"ID poison");ck(cudaMemset(out.p,0x7f,out.size*4),"out poison");
    }
};
template<class T> void equal(const Buffer<T>& a,const Buffer<T>& b,const char* name) {
    const auto x=a.download(),y=b.download();
    if(std::memcmp(x.data(),y.data(),x.size()*sizeof(T))) {
        size_t i=0;while(i<x.size()&&!std::memcmp(&x[i],&y[i],sizeof(T)))++i;
        std::fprintf(stderr,"bit mismatch %s element=%zu\n",name,i);std::exit(1);
    }
}
using Owners=std::array<std::unique_ptr<Owner>,k::kDeviceBatchSlots>;
void legacy_target(Owners& owners,Work& w,const k::DeviceWindowLayout& d,
                   Buffer<float>& gamma,const k::RopeScaling& rope,cudaStream_t stream) {
    for(uint32_t i=0;i<d.segments;++i) {
        const auto&s=d.segment[i];auto&o=*owners[s.slot];
        ck(cudaMemcpyAsync(o.snap.p,o.tail.p,384*4,cudaMemcpyDeviceToDevice,stream),"tail snapshot");
        k::kv_append_q4_steps(o.kp.p,o.vp.p,o.page.p,w.steps.p+size_t(s.first)*k::kStepCount,
            k::kStepCount,s.length,w.key.p+size_t(s.first)*512,w.value.p+size_t(s.first)*512,Shape,stream,&o.q.host);
        k::QsaIndexerBuffers ib{o.tail.p,o.dead.p,o.pooled.p,o.idxpos.p};
        k::native_qsa_indexer_append_steps(w.raw.p+s.first*128,w.steps.p+s.first*k::kStepCount+k::kStepPos,
            k::kStepCount,s.length,0,gamma.p,1e-6f,ib,Shape,Cells,rope,stream);
        k::qsa_block_scores(o.pooled.p,o.dead.p,w.queries.p+s.first*512,w.steps.p+s.first*k::kStepCount,
            s.length,MaxBlocks,Shape,w.scores.p+size_t(s.first)*MaxBlocks,stream);
        k::qsa_block_topk(w.scores.p+size_t(s.first)*MaxBlocks,w.steps.p+s.first*k::kStepCount,
            s.length,MaxBlocks,Cap,Shape,w.ids.p+size_t(s.first)*Cap,stream);
        k::kv_stream_resolve(o.q.map,o.q.pools.read_only(),o.q.host,k::kKvQ4,w.ids.p+size_t(s.first)*Cap,
            w.steps.p+s.first*k::kStepCount,s.length,Cap,Shape,stream);
        k::qsa_decode_attn_batch(w.query.p+s.first*24*256,o.q.pools.read_only(),w.ids.p+size_t(s.first)*Cap,
            w.steps.p+s.first*k::kStepCount,Cap,Shape,w.scratch.p+s.first*Stride,w.out.p+s.first*24*256,s.length,stream);
    }
}
void table_target(k::DeviceBatchView view,Work& w,int rows,int cut,Buffer<float>& gamma,
                  const k::RopeScaling& rope,cudaStream_t stream) {
    const int bound[3]={0,cut?cut:rows,rows};
    for(int group=0;group<(cut?2:1);++group) {
        const int a=bound[group],b=bound[group+1];
        k::batch_device_kv_append(view,0,a,b,w.steps.p,w.key.p,w.value.p,Shape,stream);
        k::batch_device_indexer(view,0,a,b,w.raw.p,w.steps.p,gamma.p,1e-6f,Shape,rope,false,stream);
        k::batch_device_scores(view,0,a,b,w.queries.p,w.steps.p,MaxBlocks,w.scores.p,Shape,stream);
        k::qsa_block_topk(w.scores.p+size_t(a)*MaxBlocks,w.steps.p+a*k::kStepCount,b-a,MaxBlocks,Cap,Shape,w.ids.p+size_t(a)*Cap,stream);
        k::batch_device_kv_resolve(view,0,a,b,w.ids.p,w.steps.p,Cap,Shape,stream);
        k::batch_device_attention(view,0,a,b,w.query.p,w.ids.p,w.steps.p,Cap,Shape,w.scratch.p,w.out.p,stream);
    }
}
// C4 (4 owners, 12 rows) or C8-3 (8 owners, 16 rows): the same production ABI and kernels.
void run(int owners,int rows,std::vector<std::vector<std::pair<int,int>>> cases) {
    Owners ref,got;
    for(int b=0;b<owners;++b){ref[b]=std::make_unique<Owner>(b);got[b]=std::make_unique<Owner>(b);}
    Work a,b;Buffer<float> gamma{128};gamma.upload(std::vector<float>(128,1.0f));
    Buffer<k::DeviceSlotTable> table{1};Buffer<k::DeviceWindowLayout> window{1};
    Buffer<k::DeviceCommitLayout> commit{1};Buffer<uint64_t> epoch{1};Buffer<uint32_t> status{1};epoch.object(1);
    k::DeviceSlotTable host{};for(int s=0;s<owners;++s){host.target[s].address_epoch=1;host.qsa[0][s]=got[s]->q;}table.object(host);
    k::DeviceBatchView view{table.p,window.p,commit.p,epoch.p,status.p,owners,rows};
    cudaStream_t stream{};ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"stream");
    const k::RopeScaling rope{};
    for(int width=1;width<=8;++width) cases.push_back({{width%owners,width}});
    Buffer<int32_t> warm_ids{8},warm_steps{k::kStepCount};
    std::map<std::pair<int,int>,cudaGraphExec_t> graphs,append_graphs;
    int fixtures=0,replays=0,resident_rows=0,zero_rows=0,terminal_rows=0,width_calls[9]{};
    auto capture=[&](cudaGraphExec_t& graph,auto fn) {
        cudaGraph_t raw{};ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal),"capture");fn();
        ck(cudaStreamEndCapture(stream,&raw),"end capture");ck(cudaGraphInstantiate(&graph,raw,0),"instantiate");ck(cudaGraphDestroy(raw),"graph destroy");
    };
    for(int mode=0;mode<5;++mode)for(const auto& layout:cases) {
        int owner[Rows]{};int64_t positions[Rows]{};int count=0,keep[k::kDeviceBatchSlots]{};
        for(const auto& [slot,len]:layout) {
            const int start=mode==0?0:mode==1?31+slot*7:mode==2?2051-len:
                            mode==3?258000+slot*17:262142-len;
            for(int j=0;j<len;++j,++count){owner[count]=slot;positions[count]=start+j;}
        }
        const int cut=c::segment_group_boundary(owner,count,true,true);
        const auto key=std::make_pair(count,cut);
        for(int s=0;s<owners;++s){ref[s]->reset(stream);got[s]->reset(stream);}
        ck(cudaDeviceSynchronize(),"independent fixture reset");
        // A second window retains the first window's committed state and map.
        // Near-end cases stop before a follow could exceed the logical capacity.
        for(int phase=0;phase<(mode==4?1:2);++phase) {
            std::vector<int32_t> steps(Rows*k::kStepCount);
            for(int r=0;r<count;++r)k::qsa_step_fill(steps.data()+r*k::kStepCount,positions[r],Shape);
            k::DeviceWindowLayout d;std::string error;
            require(c::make_device_window(owner,positions,count,cut,{1,2,3,4,5,6,7,8},{1,1,1,1,1,1,1,1},fixtures+1,1,1,d,
                                          error,owners,rows),error.c_str());
            for(uint32_t i=0;i<d.segments;++i)keep[d.segment[i].slot]=1+(fixtures+i)%d.segment[i].length;
            k::DeviceCommitLayout cd;require(c::make_device_commit(d,keep,cd,error,owners,rows),error.c_str());
            window.object(d);commit.object(cd);status.zero();a.inputs(0x1234+fixtures,steps);b.inputs(0x1234+fixtures,steps);
            ck(cudaDeviceSynchronize(),"descriptor and input ready");
            if(phase) {
                // Pre-resolve every future append page, then test append alone
                // before any later resolve can conceal a missing pool mirror.
                for(uint32_t i=0;i<d.segments;++i) {
                    const auto& s=d.segment[i];std::vector<int32_t> ids(8),st(k::kStepCount);
                    for(int j=0;j<s.length;++j)ids[j]=(int32_t)positions[s.first+j];
                    st[k::kStepWidth]=s.length;warm_ids.upload(ids);warm_steps.upload(st);
                    ck(cudaDeviceSynchronize(),"warm IDs ready");
                    for(Owner* o:{ref[s.slot].get(),got[s.slot].get()})
                        k::kv_stream_resolve(o->q.map,o->q.pools.read_only(),o->q.host,k::kKvQ4,warm_ids.p,warm_steps.p,1,8,Shape,stream);
                    ck(cudaStreamSynchronize(stream),"future pages resident");
                    auto& o=*ref[s.slot];
                    k::kv_append_q4_steps(o.kp.p,o.vp.p,o.page.p,a.steps.p+s.first*k::kStepCount,k::kStepCount,
                        s.length,a.key.p+s.first*512,a.value.p+s.first*512,Shape,stream,&o.q.host);
                }
                auto& ag=append_graphs[key];
                if(!ag)capture(ag,[&]{
                    const int bounds[3]={0,cut?cut:count,count};
                    for(int g=0;g<(cut?2:1);++g)k::batch_device_kv_append(view,0,bounds[g],bounds[g+1],b.steps.p,b.key.p,b.value.p,Shape,stream);
                });
                ck(cudaGraphLaunch(ag,stream),"resident append graph");ck(cudaStreamSynchronize(stream),"resident append sync");
                require(status.download()[0]==0,"resident append descriptor status");
                for(uint32_t i=0;i<d.segments;++i) {
                    const auto& s=d.segment[i];ref[s.slot]->check_appended_pages(steps,s);got[s.slot]->check_appended_pages(steps,s);
                    resident_rows+=s.length;
                }
            }
            auto& graph=graphs[key];
            if(!graph)capture(graph,[&]{table_target(view,b,count,cut,gamma,rope,stream);});else ++replays;
            legacy_target(ref,a,d,gamma,rope,stream);ck(cudaGraphLaunch(graph,stream),"replay");ck(cudaStreamSynchronize(stream),"target sync");
            require(status.download()[0]==0,"descriptor status");equal(a.scores,b.scores,"scores+inactive poison");equal(a.ids,b.ids,"IDs+inactive poison");equal(a.out,b.out,"attention+inactive poison");
            const auto result=b.out.download();for(int i=0;i<count*24*256;++i)require(std::isfinite(result[i]),"finite attention");
            const auto ids=b.ids.download();
            for(uint32_t i=0;i<d.segments;++i){ref[d.segment[i].slot]->check_pages(ids,steps,d.segment[i]);got[d.segment[i].slot]->check_pages(ids,steps,d.segment[i]);}
            for(int s=0;s<owners;++s) {
                ref[s]->check_map();got[s]->check_map();
                const auto x=ref[s]->ctl.download(),y=got[s]->ctl.download();
                require(x[0]==y[0] && x[2]==y[2] && x[3]==0 && y[3]==0,"resolver epoch/misses/no overflow");
                for(int i=4;i<10;++i)require(x[i]==y[i],"integer resolver miss/lookup/call totals");
                equal(ref[s]->tail,got[s]->tail,"target tail owner");equal(ref[s]->dead,got[s]->dead,"target dead owner");
                equal(ref[s]->pooled,got[s]->pooled,"target pooled owner");equal(ref[s]->idxpos,got[s]->idxpos,"target block position owner");
                require(!std::memcmp(ref[s]->hk.p,got[s]->hk.p,ref[s]->hk.size),"all Host K including inactive owners");
                require(!std::memcmp(ref[s]->hv.p,got[s]->hv.p,ref[s]->hv.size),"all Host V including inactive owners");
            }
            for(uint32_t i=0;i<d.segments;++i){const auto&s=d.segment[i];auto&o=*ref[s.slot];
                ck(cudaMemcpyAsync(o.tail.p,o.snap.p,384*4,cudaMemcpyDeviceToDevice,stream),"restore tail");
                k::QsaIndexerBuffers ib{o.tail.p,o.dead.p,o.pooled.p,o.idxpos.p};
                k::native_qsa_indexer_append_steps(a.raw.p+s.first*128,a.steps.p+s.first*k::kStepCount+k::kStepPos,
                    k::kStepCount,keep[s.slot],0,gamma.p,1e-6f,ib,Shape,Cells,rope,stream);
            }
            k::batch_device_indexer(view,0,0,count,b.raw.p,b.steps.p,gamma.p,1e-6f,Shape,rope,true,stream);
            ck(cudaStreamSynchronize(stream),"commit sync");require(status.download()[0]==0,"commit status");
            for(int s=0;s<owners;++s){equal(ref[s]->tail,got[s]->tail,"commit tail owner");equal(ref[s]->dead,got[s]->dead,"commit dead owner");equal(ref[s]->pooled,got[s]->pooled,"commit pooled owner");equal(ref[s]->idxpos,got[s]->idxpos,"commit block position owner");}
            for(uint32_t i=0;i<d.segments;++i) {
                const auto& s=d.segment[i];++width_calls[s.length];
                for(int r=s.first;r<s.first+s.length;++r){zero_rows+=positions[r]==0;terminal_rows+=positions[r]==262141;}
                const int64_t next=positions[s.first]+keep[s.slot];
                for(int j=0;j<s.length;++j)positions[s.first+j]=next+j;
            }
            ++fixtures;
        }
    }
    for(auto& [key,g]:graphs)ck(cudaGraphExecDestroy(g),"exec destroy");
    for(auto& [key,g]:append_graphs)ck(cudaGraphExecDestroy(g),"append exec destroy");
    ck(cudaStreamDestroy(stream),"stream destroy");
    require(zero_rows>0&&terminal_rows>0&&resident_rows>0,"boundary and resident fixtures executed");
    for(int w=1;w<=8;++w)require(width_calls[w]>0,"all causal widths executed");
    std::printf("PASS table Q4 owners=%d rows=%d pooled=%s: %d windows, %zu target + %zu resident-append rows-cut graph families, %d target replays; resident_rows=%d pos0=%d terminal262141=%d; full logical256K, no weights\n",
        owners,rows,k::qsa_idx_pooled_bf16()?"bf16":"fp32",fixtures,graphs.size(),append_graphs.size(),replays,resident_rows,
        zero_rows,terminal_rows);
    for(int w=1;w<=8;++w)std::printf("width=%d windows=%d\n",w,width_calls[w]);
}
int main() {
#if defined(STRATA_USE_HIP) || defined(STRATA_GPU_METAL)
    return 77;
#endif
    int devices=0;
    if(cudaGetDeviceCount(&devices)!=cudaSuccess || devices<1) return 77;
    size_t free_bytes=0,total_bytes=0;ck(cudaMemGetInfo(&free_bytes,&total_bytes),"memory info");
    if(free_bytes<(8ull<<30)+(1024ull<<20)) {
        std::fprintf(stderr,"SKIP: preserving 8 GiB reserve with 1 GiB fixture allowance\n");return 77;
    }
    require(k::batch_device_scores_supported(),"normal score profile required");
    run(4,12,{{{0,6},{3,6}},{{3,3},{1,3},{0,3},{2,3}},
        {{1,6},{2,6}},{{2,3},{0,3},{3,3},{1,3}},{{0,4},{2,4},{3,4}},{{3,8},{1,4}},{{2,4},{0,8}}});
    // C8-3: A8 2 rows each, A4 4 each on owners 4..7, A3 5+4+4, A2 6+6, mixed widths, 8+8 and owners past 3
    const std::vector<std::vector<std::pair<int,int>>> c8={{{7,2},{0,2},{6,2},{1,2},{5,2},{2,2},{4,2},{3,2}},
        {{4,4},{5,4},{6,4},{7,4}},{{6,5},{2,4},{5,4}},{{1,6},{6,6}},{{5,3},{1,3},{7,2},{0,2},{2,2},{4,2},{6,2}},
        {{6,8},{3,8}},{{3,4},{6,4},{0,4},{5,4}},{{2,1},{7,1},{0,1},{5,1},{3,1},{6,1},{1,1},{4,1}}};
    run(8,16,c8);
    // C8-6: the same table == legacy contract with the pooled store in BF16 (both sides write and score it through
    // the same mode-dispatched kernels; idx_pooled_gpu holds BF16 against FP32 itself)
    k::qsa_set_idx_pooled_bf16(true);
    run(8,16,c8);
    k::qsa_set_idx_pooled_bf16(false);
}
