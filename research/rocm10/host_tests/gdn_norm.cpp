#include "gdn_norm.hpp"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
using namespace strixlab_gdn_norm;
void require(bool ok,const char* message){if(!ok){std::cerr<<message<<'\n';std::abort();}}
uint32_t bits(float x){uint32_t u;std::memcpy(&u,&x,4);return u;}
// Independent host reference of the exact 256-thread reduction DAG, rather
// than calling the implementation's warp_sum/block_sum. Pairing is XOR in
// each 32-lane wave, followed by the 8 wave leaders plus 24 zero lanes.
float reference_sum(const float* x){
    float a[256]{};
    for(int i=0;i<128;++i)a[i]=x[i]*x[i]+0.0f;
    for(int offset:{16,8,4,2,1}){
        float old[256];std::copy(a,a+256,old);
        for(int i=0;i<256;++i)a[i]=old[i]+old[(i/32)*32+((i%32)^offset)];
    }
    float b[32]{};for(int i=0;i<8;++i)b[i]=a[i*32];
    for(int offset:{16,8,4,2,1}){
        float old[32];std::copy(b,b+32,old);
        for(int i=0;i<32;++i)b[i]=old[i]+old[i^offset];
    }
    return b[0];
}
float vector_value(int kind,int col){
    const float sign=col%2?-1.0f:1.0f;
    switch(kind%8){
        case 0:return col%2?-0.0f:0.0f;
        case 1:return sign*std::sqrt(1e-6f/128.0f); // sum squares near epsilon
        case 2:return sign*1e-18f;
        case 3:return sign*1e15f; // finite squared sum (no overflow)
        case 4:return sign*float(col+1)/31.0f;
        case 5:return col==31?0.01f:(col==32?-0.02f:1e-9f);
        case 6:return col%3?sign*1e-4f:sign*1e3f;
        default:return sign*1e-30f; // square underflow; epsilon dominates
    }
}
void correctness(Shape s,bool independent_strides){
    const int64_t hs=independent_strides?132:128;
    const int64_t ts=independent_strides?8400:8192;
    const int64_t ss=ts*s.tokens+256;
    const int64_t k_offset=independent_strides?4096:2048;
    std::vector<float> input(size_t(ss)*s.sequences+2,9876.0f);
    Input q{input.data()+1,hs,ts,ss,1e-6f};
    Input k{input.data()+1+k_offset,hs,ts,ss,4e-6f};
    for(int b=0;b<s.sequences;++b)for(int t=0;t<s.tokens;++t)for(int h=0;h<s.heads;++h)
        for(int c=0;c<128;++c){
            const auto offset=b*ss+t*ts+h*hs+c;
            input[1+offset]=vector_value(h+2*t+3*b,c);
            input[1+k_offset+offset]=vector_value(h+2*t+3*b+1,c);
        }
    const auto before=input;
    const size_t n=elements(s);
    std::vector<float> qo(n+2,7654),ko(n+2,7654),fq(n+2,7654),fk(n+2,7654),scratch(2*n+2,7654);
    host_launches.clear();
    require(baseline(q,k,qo.data()+1,ko.data()+1,scratch.data()+1,s,nullptr)==hipSuccess,"baseline launch");
    require(host_launches.size()==4,"four baseline launches");
    for(int i:{0,2})require(host_launches[i].grid.x==unsigned(s.heads)&&host_launches[i].grid.y==unsigned(s.tokens)&&host_launches[i].grid.z==unsigned(s.sequences)&&host_launches[i].shared==128,"baseline RMS grid/shared");
    require(fused(q,k,fq.data()+1,fk.data()+1,s,nullptr)==hipSuccess,"fused launch");
    require(host_launches.size()==5 && host_launches.back().grid.z==unsigned(2*s.sequences),"one fused launch");
    for(auto& v:{&qo,&ko,&fq,&fk,&scratch})require(v->front()==7654&&v->back()==7654,"canaries");
    require(std::memcmp(before.data(),input.data(),input.size()*4)==0,"input preservation");
    bool detects_old_epsilon=false,detects_algebraic_collapse=false;
    for(int which=0;which<2;++which){
        const Input in=which?k:q;const auto& out=which?ko:qo;const auto& candidate=which?fk:fq;
        for(int b=0;b<s.sequences;++b)for(int t=0;t<s.tokens;++t)for(int h=0;h<s.heads;++h){
            const float* x=in.data+b*in.stride_sequence+t*in.stride_token+h*in.stride_head;
            const float sum=reference_sum(x),mean=sum/128.0f;
            const float scale=1.0f/std::sqrt(mean+in.eps/128.0f);
            double sum64=0;for(int c=0;c<128;++c)sum64+=double(x[c])*double(x[c]);
            for(int c=0;c<128;++c){
                const size_t i=1+((b*s.tokens+t)*s.heads+h)*128+c;
                volatile float normalized=scale*x[c];
                const float expected=(1.0f/std::sqrt(128.0f))*normalized+0.0f;
                const double oracle=double(x[c])/std::sqrt(sum64+double(in.eps));
                const float old_l2=x[c]/std::sqrt(std::max(sum,in.eps*in.eps));
                const float collapsed=x[c]/std::sqrt(sum+in.eps);
                detects_old_epsilon|=bits(expected)!=bits(old_l2);
                detects_algebraic_collapse|=bits(expected)!=bits(collapsed);
                require(bits(out[i])==bits(candidate[i]),"bitwise host baseline/candidate parity");
                require(bits(out[i])==bits(expected),"independent reduction DAG/rounding");
                require(std::isfinite(out[i])&&std::abs(double(out[i])-oracle)<=2e-6+2e-5*std::abs(oracle),"corrected double oracle");
                if(x[c]==0)require(bits(out[i])==0,"SCALE +0 signed zero");
            }
        }
    }
    require(detects_old_epsilon,"vectors expose old epsilon formula");
    require(detects_algebraic_collapse,"vectors expose collapsed FP32 formula");
}
void validation(){
    Shape s;std::vector<float> input(8192),qo(elements(s)),ko(elements(s)),scratch(2*elements(s));
    Input q{input.data(),128,8192,8192,1e-6f},k{input.data()+2048,128,8192,8192,2e-6f};
    host_record_only=true;host_launches.clear();
    auto bad=[&](Input a,Input b,float* x,float* y,float* tmp,Shape shape){
        const auto n=host_launches.size();
        require(baseline(a,b,x,y,tmp,shape,nullptr)==hipErrorInvalidValue,"reject baseline invalid");
        require(fused(a,b,x,y,shape,nullptr)==hipErrorInvalidValue,"reject fused invalid");
        require(host_launches.size()==n,"validation no launch");
    };
    for(Shape shape: {Shape{127,16,1,1},Shape{128,0,1,1},Shape{128,16,0,1},Shape{128,16,1,32768}})bad(q,k,qo.data(),ko.data(),scratch.data(),shape);
    for(float eps:{0.0f,-1.0f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::denorm_min()}){Input a=q;a.eps=eps;bad(a,k,qo.data(),ko.data(),scratch.data(),s);}
    for(int64_t stride:{int64_t(-1),int64_t(127),std::numeric_limits<int64_t>::max()}){Input a=q;a.stride_head=stride;bad(a,k,qo.data(),ko.data(),scratch.data(),s);}
    Input a=q;a.data=nullptr;bad(a,k,qo.data(),ko.data(),scratch.data(),s);
    bad(q,k,nullptr,ko.data(),scratch.data(),s);
    bad(q,k,qo.data(),qo.data()+1,scratch.data(),s);
    bad(q,k,input.data()+1,ko.data(),scratch.data(),s);
    require(baseline(q,k,qo.data(),ko.data(),qo.data(),s,nullptr)==hipErrorInvalidValue,"scratch alias");
    require(baseline(q,k,qo.data(),ko.data(),nullptr,s,nullptr)==hipErrorInvalidValue,"null scratch");
    for(size_t fail=1;fail<=4;++fail){host_launches.clear();host_fail_at=fail;require(baseline(q,k,qo.data(),ko.data(),scratch.data(),s,nullptr)==73,"launch error propagated");require(host_launches.size()==fail,"stop on first error");}
    host_launches.clear();host_fail_at=1;require(fused(q,k,qo.data(),ko.data(),s,nullptr)==73,"fused error propagated");host_fail_at=0;
    for(int tokens:{1,16,512}){
        Shape shape{128,16,tokens,1};
        require(elements(shape)==size_t(128*16*tokens)&&scratch_bytes(shape)==size_t(2*128*16*tokens*4),"production shape counts");
        std::vector<float> src(size_t(8192)*tokens),out(4*elements(shape)),tmp(2*elements(shape));
        Input a{src.data(),128,8192,int64_t(8192)*tokens,1e-6f},b=a;b.data+=2048;
        const size_t start=host_launches.size();
        require(baseline(a,b,out.data(),out.data()+elements(shape),tmp.data(),shape,nullptr)==hipSuccess,"production baseline layout accepted");
        require(fused(a,b,out.data()+2*elements(shape),out.data()+3*elements(shape),shape,nullptr)==hipSuccess,"production fused layout accepted");
        require(host_launches.size()==start+5&&host_launches.back().grid.x==16&&host_launches.back().grid.y==unsigned(tokens),"production dispatch recording");
    }
    host_record_only=false;
}
int main(int argc,char** argv){
    if(argc==2&&std::string(argv[1])=="wrong-wave"){
        warpSize=64;Shape s{128,1,1,1};std::vector<float> q(128),k(128),qo(128),ko(128);
        fused({q.data(),128,128,128,1e-6f},{k.data(),128,128,128,1e-6f},qo.data(),ko.data(),s,nullptr);return 0;
    }
    validation();correctness({128,16,1,1},false);correctness({128,2,2,2},true);
    std::cout<<"PASS: 48 Q/K rows; source reduction DAG, four-to-one dispatch, signed zero, corrected epsilon, strides, canaries, aliases, launch errors (CPU emulation only)\n";
}
