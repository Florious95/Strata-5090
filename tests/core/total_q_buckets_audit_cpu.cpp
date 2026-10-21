#include "strata/core/window_audit.hpp"
#include <filesystem>
#include <fstream>
#include <iterator>
#include <cstdio>
#include <cstdlib>

namespace c = strata::core;
static int checks = 0;
static void check(bool ok,const char* what) {
    ++checks; if (!ok) { std::fprintf(stderr,"FAIL total-Q audit: %s\n",what); std::exit(1); }
}
static c::AuditConfig config(bool enabled) {
    c::AuditConfig c;
    c.slots=4; c.max_rows=12; c.max_context=262144; c.resident_requested=20480;
    c.effective_resident={20480,20480,20480,20480};
    c.generalized=c.canonical=c.segmented=c.row_table=true; c.total_q_buckets=enabled;
    return c;
}
static std::string path() {
    static int suffix=0;
    return (std::filesystem::temp_directory_path()/("strata-total-q-audit-"+
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+"-"+
        std::to_string(++suffix)+".jsonl")).string();
}
static c::AuditWindow window(c::WindowAudit& writer,int id) {
    const auto& t=c::kTotalQTemplates[id]; c::AuditWindow w;
    for (int b=0;b<4;++b) if (t.witness[b]) {
        check(writer.begin_admission(b) && writer.end_admission(0,true,nullptr,true),"owned admission");
        w.active.push_back(b);
        c::AuditSlot s; s.id=b;s.generation=writer.generation(b);s.rows=s.useful=t.witness[b];
        s.keep=1;s.emitted=1;s.position=258000;w.slots.push_back(s);
        for(int j=0;j<s.rows;++j) w.row_slots.push_back(b);
    }
    w.route={t.rows*480,t.rows*480,0,0,0};
    auto& d=w.dispatch;
    d.canonical=d.generalized=d.merged=d.dense_parallel=d.shared_after_ring=d.graph_hit=true;
    d.dense_groups=2;d.execution_path="batch_rows";d.graph_key_kind="rows_cut";
    d.graph_key_rows=t.rows;d.graph_key_cut=t.cut;d.graph_key_profile=286;
    d.total_q_template=id;d.total_q_bucket=t.bucket;d.total_q_offered_rows=t.rows;
    return w;
}
int main() {
    for (int id=0;id<(int)c::kTotalQTemplates.size();++id) {
        const auto p=path(); c::WindowAudit writer;
        check(writer.open(p.c_str(),config(true)),"enabled header");
        auto w=window(writer,id);
        check(writer.window(w),"all eight exact G2 templates serialize");
        for (int b:w.active) check(writer.finish_slot(b,"length"),"finish owner");
        check(writer.close(),"valid footer");
        std::ifstream f(p);const std::string s((std::istreambuf_iterator<char>(f)),{});
        check(s.find("\"total_q_buckets\":true")!=std::string::npos,"header capability");
        check(s.find("\"total_q_template_pairs\":8")!=std::string::npos,"honest eight-pair bound");
        check(s.find("\"row_table_bucket\":{\"family\":"+std::to_string(c::kTotalQTemplates[id].bucket))!=std::string::npos,"bucket family telemetry");
        check(s.find("\"offered_rows\":"+std::to_string(c::kTotalQTemplates[id].rows))!=std::string::npos,"lost-work denominator");
        check(s.find("\"invalid_records\":0")!=std::string::npos,"complete lifecycle");
        f.close();std::filesystem::remove(p);
    }
    // Each case starts with a known-valid complete owner lifecycle prefix.
    for (int mutation=0;mutation<10;++mutation) {
        const auto p=path();c::WindowAudit writer;
        check(writer.open(p.c_str(),config(true)),"negative header");
        auto w=window(writer,6);auto& d=w.dispatch;
        if(mutation==0)d.graph_hit=false;
        if(mutation==1)d.dense_parallel=false;
        if(mutation==2)d.shared_after_ring=false;
        if(mutation==3)d.graph_key_cut=5;
        if(mutation==4)d.total_q_template=-1;
        if(mutation==5)d.total_q_bucket=9;
        if(mutation==6)d.total_q_offered_rows=11;
        if(mutation==7)d.total_q_offered_rows=13;
        if(mutation==8)d.execution_path="legacy";
        if(mutation==9)d.dense_groups=1;
        check(!writer.window(w) && !writer.good(),"miss/serialized/corrupt template fails closed");
        check(!writer.close(),"failure is not a successful footer");
        std::filesystem::remove(p);
    }
    {
        const auto p=path();c::WindowAudit writer;
        check(writer.open(p.c_str(),config(false)),"OFF compatibility header");
        auto w=window(writer,6);w.dispatch.total_q_template=-1;w.dispatch.total_q_bucket=0;
        w.dispatch.total_q_offered_rows=0;w.dispatch.graph_hit=false;
        w.dispatch.dense_parallel=false;w.dispatch.shared_after_ring=false;
        check(writer.window(w),"OFF keeps existing capture/serial behavior");
        for(int b:w.active) check(writer.finish_slot(b,"length"),"OFF finish");
        check(writer.close(),"OFF footer");std::filesystem::remove(p);
    }
    {
        const auto p=path();c::WindowAudit writer;auto c=config(true);c.slots=2;c.effective_resident={20480,20480};
        check(!writer.open(p.c_str(),c),"reject non-C4 capability");
        check(!std::filesystem::exists(p),"invalid header creates no file");
    }
    std::printf("PASS total-Q audit: %d checks, 8 serialized exact G2 templates, 10 fail-closed mutations, OFF compatibility. CPU only.\n",checks);
}
