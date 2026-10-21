"""Public Streamlit application using precisely the native Transformer generator."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT/'transformer'/'src'):
    if str(folder) not in sys.path: sys.path.insert(0,str(folder))
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from cwt_processing.synthetic_peak_generator import SyntheticPeakDataset
from web_generator.bridge import generate_chunk,combine_chunks,zip_dataset
from web_generator.wavelet_preview import WAVELET_TYPES,wavelet_transform_1d

st.set_page_config(page_title='Double Peak Generator',layout='wide')
st.title('Double Peak Generator')
st.caption('Publication edition — same centered synthetic generator as CWT Transformer; Gaussian, Lorentzian, pseudo-Voigt and mixed families.')

DEFAULT=dict(family='gaussian',n_samples=1000,seed=31337,n_points=400,x_min=0.,x_max=400.,
             center_mode='fixed',delta_mu_min=10.,delta_mu_max=None,
             sigma_min_frac=0.02,sigma_max_frac=0.06,sigma_ratio_min=0.5,sigma_ratio_max=2.,
             delta_width_min=0.25,delta_width_max=3.,edge_sigma_margin=4.,
             amplitude_high_min=0.5,amplitude_high_max=1.0,
             amplitude_ratio_min=0.1,amplitude_ratio_max=1.0,
             eta_min=0.,eta_max=1.,noise_level=0.,distinct_epochs=False)
# These are the original experiment-engine defaults. N=1000 is chosen for browser use;
# the 50,000-sample training command is documented in transformer/README.md.

def reset():
    for k in list(DEFAULT):st.session_state.pop('p_'+k,None)
    for k in ('p_delta_on','p_delta_limit','p_noise_percent','generated','example_ids'):
        st.session_state.pop(k,None)

left,right=st.columns([0.43,0.57],gap='large')
with left:
    st.subheader('Generator settings')
    st.button('Restore publication defaults',on_click=reset)
    with st.form('generator_controls'):
        family=st.selectbox('Peak family',['gaussian','lorentzian','pseudovoigt','paired_mix','independent_mix'],key='p_family',
            help='paired_mix = a common randomly selected family; independent_mix = independently selected families.')
        c1,c2=st.columns(2)
        with c1:n_samples=st.number_input('Samples',min_value=1,max_value=10000,value=1000,step=100,key='p_n_samples')
        with c2:seed=st.number_input('Seed',min_value=0,value=31337,step=1,key='p_seed')
        st.markdown('**Axis and centering**')
        c1,c2,c3=st.columns(3)
        with c1:x_min=st.number_input('X min',value=0.,key='p_x_min')
        with c2:x_max=st.number_input('X max',value=400.,key='p_x_max')
        with c3:n_points=st.number_input('Points',min_value=32,max_value=4096,value=400,key='p_n_points')
        center_mode=st.selectbox('Pair center',['fixed','random'],key='p_center_mode')
        st.caption('Fixed center = midpoint of the selected axis (publication default: 200).')
        with st.expander('Peak spacing and widths',expanded=True):
            delta_min=st.number_input('Minimum Δμ [axis units]',min_value=0.,value=10.,key='p_delta_mu_min')
            delta_on=st.checkbox('Limit maximum Δμ',value=False,key='p_delta_on')
            delta_max=st.number_input('Maximum Δμ',min_value=0.,value=100.,key='p_delta_limit') if delta_on else None
            c1,c2=st.columns(2)
            with c1:
                sig_lo=st.number_input('σ/span minimum',min_value=0.001,max_value=0.5,value=0.02,format='%.3f',key='p_sigma_min_frac')
                ratio_lo=st.number_input('σ1/σ2 minimum',min_value=0.01,max_value=1.,value=0.5,key='p_sigma_ratio_min')
                dwidth_lo=st.number_input('Δμ/FWHM minimum',min_value=0.,value=0.25,key='p_delta_width_min')
            with c2:
                sig_hi=st.number_input('σ/span maximum',min_value=0.001,max_value=0.5,value=0.06,format='%.3f',key='p_sigma_max_frac')
                ratio_hi=st.number_input('σ1/σ2 maximum',min_value=1.,value=2.,key='p_sigma_ratio_max')
                dwidth_hi=st.number_input('Δμ/FWHM maximum',min_value=0.01,value=3.,key='p_delta_width_max')
            margin=st.number_input('Edge margin [σ]',min_value=0.,value=4.,key='p_edge_sigma_margin')
        with st.expander('Amplitude and pseudo-Voigt η',expanded=True):
            c1,c2=st.columns(2)
            with c1:
                ah_lo=st.number_input('Max(A) minimum',min_value=0.001,value=0.5,key='p_amplitude_high_min')
                ar_lo=st.number_input('A ratio minimum',min_value=0.001,max_value=1.,value=0.1,key='p_amplitude_ratio_min')
                eta_lo=st.number_input('η minimum',min_value=0.,max_value=1.,value=0.,key='p_eta_min')
            with c2:
                ah_hi=st.number_input('Max(A) maximum',min_value=0.001,value=1.,key='p_amplitude_high_max')
                ar_hi=st.number_input('A ratio maximum',min_value=0.001,max_value=1.,value=1.,key='p_amplitude_ratio_max')
                eta_hi=st.number_input('η maximum',min_value=0.,max_value=1.,value=1.,key='p_eta_max')
        noise_pc=st.number_input('Gaussian noise σ [% max of summed signal]',min_value=0.,max_value=100.,value=0.,step=0.1,format='%.2f',key='p_noise_percent')
        st.caption('Native convention: σ_noise = noise_level × max(summed signal). The old per-component noise rule is not used.')
        clicked=st.form_submit_button('Generate data',type='primary',use_container_width=True)
    if clicked:
        config={**DEFAULT,'family':family,'n_samples':int(n_samples),'seed':int(seed),
                'n_points':int(n_points),'x_min':float(x_min),'x_max':float(x_max),
                'center_mode':center_mode,'delta_mu_min':float(delta_min),'delta_mu_max':delta_max,
                'sigma_min_frac':sig_lo,'sigma_max_frac':sig_hi,
                'sigma_ratio_min':ratio_lo,'sigma_ratio_max':ratio_hi,
                'delta_width_min':dwidth_lo,'delta_width_max':dwidth_hi,
                'edge_sigma_margin':margin,'amplitude_high_min':ah_lo,'amplitude_high_max':ah_hi,
                'amplitude_ratio_min':ar_lo,'amplitude_ratio_max':ar_hi,
                'eta_min':eta_lo,'eta_max':eta_hi,'noise_level':noise_pc/100.}
        try:
            native=SyntheticPeakDataset(**config)
            bar=st.progress(0,text='Generating')
            parts=[]
            for i in range(0,len(native),250):
                parts.append(generate_chunk(native,i,min(i+250,len(native))))
                bar.progress(min(1.,(i+250)/len(native)),text='Generating')
            st.session_state.generated=combine_chunks(parts)
            st.session_state.pop('example_ids',None)
            bar.empty()
            st.success(f'{len(native)} synthetic spectra generated using the Transformer generator.')
        except Exception as exc:
            st.error(f'Generation failed: {exc}')

with right:
    st.subheader('Explore and export')
    result=st.session_state.get('generated')
    if result is None:
        st.info('Configure the peaks and select **Generate data**. No trained weights or checkpoint are needed for generation.')
    else:
        st.caption(f'{len(result.summed):,} samples × {len(result.x):,} points · centered={result.config["center_mode"]} · seed={result.config["seed"]}')
        name=st.text_input('File name',value='centered_peaks')
        st.download_button('Download ZIP · spectra / labels / config',zip_dataset(result,name),
                           file_name=f'{name or "centered_peaks"}.zip',mime='application/zip',use_container_width=True)
        st.download_button('Download labels CSV',result.labels.to_csv(index=False).encode('utf-8'),
                           file_name=f'{name or "centered_peaks"}_labels.csv',mime='text/csv')
        st.subheader('Signal preview')
        n=min(6,len(result.summed))
        if 'example_ids' not in st.session_state:
            rng=np.random.default_rng(result.config['seed']+999)
            st.session_state.example_ids=sorted(map(int,rng.choice(len(result.summed),n,replace=False)))
        if st.button('Draw another six'):
            st.session_state.example_ids=sorted(map(int,np.random.default_rng().choice(len(result.summed),n,replace=False)))
        c1,c2,c3=st.columns(3)
        with c1:wave=st.selectbox('Wavelet preview',['None',*WAVELET_TYPES])
        with c2:scale=st.number_input('Scale (points)',min_value=0.1,value=5.,step=1.)
        with c3:source=st.selectbox('Transform',['Sum','Peak 1','Peak 2'])
        normalize=st.checkbox('Normalize preview overlay',value=True)
        for i in st.session_state.example_ids:
            fig,ax=plt.subplots(figsize=(8.1,2.6))
            ax.plot(result.x,result.summed[i],color='#172554',label='Sum',lw=1.8)
            ax.plot(result.x,result.peak1[i],color='#0f766e',label='Peak 1',alpha=.85)
            ax.plot(result.x,result.peak2[i],color='#e08e21',label='Peak 2',alpha=.85)
            if wave!='None':
                s={'Sum':result.summed,'Peak 1':result.peak1,'Peak 2':result.peak2}[source][i]
                try:
                    w=wavelet_transform_1d(s,wave,scale)
                    if normalize:w=w/max(np.max(np.abs(w)),1e-12)*max(np.max(np.abs(s)),1e-12)
                    ax.plot(result.x,w,'--',color='#be185d',label=f'{wave}@{scale:g}')
                except Exception as err:st.warning(f'Preview error: {err}')
            ax.set(xlabel='X',ylabel='Amplitude',title=f'Sample {i}')
            ax.legend(loc='best',fontsize=7,ncol=2)
            st.pyplot(fig,clear_figure=True)
            plt.close(fig)
        st.dataframe(result.labels.iloc[st.session_state.example_ids],use_container_width=True,hide_index=True)

st.caption('Training baseline: 50,000 total samples, split 40,000/5,000/5,000; MH + GD, scales 5 and 23, 50 epochs. See transformer/README.md.')
