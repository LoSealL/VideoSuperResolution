# Collections Specially for CVPR 2019
*No workshop publication included*

Terms:
- SR: super-resolution
- SOTA: state-of-the-art
- DSLR: digital single-lens reflex camera

## SISR
- [Meta-SR](http://openaccess.thecvf.com/content_CVPR_2019/papers/Hu_Meta-SR_A_Magnification-Arbitrary_Network_for_Super-Resolution_CVPR_2019_paper.pdf)
   - Early published [arXiv](https://arxiv.org/abs/1903.00875) 
   - Codes: [Pytorch](https://github.com/XuecaiHu/Meta-SR-Pytorch)
   - Remark: Face++, arbitrary scale factor

- [IKC](http://openaccess.thecvf.com/content_CVPR_2019/papers/Gu_Blind_Super-Resolution_With_Iterative_Kernel_Correction_CVPR_2019_paper.pdf)
   - Early published [arXiv](https://arxiv.org/abs/1904.03377)
   - Remark: SenseTime, unknown blur kernel

- [CameraSR](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Camera_Lens_Super-Resolution_CVPR_2019_paper.pdf)
   - Remark: SR on DSLR camera

- [DPSR]()
   - Early published [arXiv](https://arxiv.org/abs/1903.12529)
   - Codes: [Pytorch](https://github.com/cszn/DPSR)
   - Remark: Jointly SR & deblur

- [Towards Real Scene Super-Resolution with Raw Images](http://openaccess.thecvf.com/content_CVPR_2019/papers/Xu_Towards_Real_Scene_Super-Resolution_With_Raw_Images_CVPR_2019_paper.pdf)
   - Early published [arXiv](https://arxiv.org/abs/1905.12156)
   - Remark: SenseTime, raw image SR
   
- [OISR](http://openaccess.thecvf.com/content_CVPR_2019/papers/He_ODE-Inspired_Network_Design_for_Single_Image_Super-Resolution_CVPR_2019_paper.pdf)
   - Remark: ODE-inspired blocks

- [SRFBN](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Feedback_Network_for_Image_Super-Resolution_CVPR_2019_paper.pdf)
   - Early published [arXiv](https://arxiv.org/abs/1903.09814)
   - Codes: [Pytorch](https://github.com/Paper99/SRFBN_CVPR19)

- [SRNTT](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Image_Super-Resolution_by_Neural_Texture_Transfer_CVPR_2019_paper.pdf)
   - Early published [arXiv](https://arxiv.org/abs/1903.00834)
   - Codes: [Tensorflow](https://github.com/ZZUTK/SRNTT)
   - Remark: Adobe, reference based SR

- [NatSR](http://openaccess.thecvf.com/content_CVPR_2019/papers/Soh_Natural_and_Realistic_Single_Image_Super-Resolution_With_Explicit_Natural_Manifold_CVPR_2019_paper.pdf)
   - Codes: [empty](https://github.com/JWSoh/NatSR)
   - Remark: GAN

- [3DASR](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_3D_Appearance_Super-Resolution_With_Deep_Learning_CVPR_2019_paper.pdf)
   - Early published [arXiv](https://arxiv.org/abs/1906.00925)
   - Codes: [Pytorch](https://github.com/ofsoundof/3D_Appearance_SR)
   - Remark: 3D, multiview, game

- [SAN](http://openaccess.thecvf.com/content_CVPR_2019/papers/Dai_Second-Order_Attention_Network_for_Single_Image_Super-Resolution_CVPR_2019_paper.pdf)
   - Codes: [empty](https://github.com/daitao/SAN)
   - Remark: Ali DAMO, 2nd-order attention, **SOTA**(nearlly)

- [Hyperspectral Image Super-Resolution with Optimized RGB Guidance](http://openaccess.thecvf.com/content_CVPR_2019/papers/Fu_Hyperspectral_Image_Super-Resolution_With_Optimized_RGB_Guidance_CVPR_2019_paper.pdf)
   - Remark: HSI (hyperspectral image)
   
- [PASSRnet](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Learning_Parallax_Attention_for_Stereo_Image_Super-Resolution_CVPR_2019_paper.pdf)
   - Early published [arXiv](https://arxiv.org/abs/1903.05784)
   - Codes: [Pytorch](https://github.com/LongguangWang/PASSRnet)
   - Remark: Stereo Image

## VSR
- [RBPN](http://openaccess.thecvf.com/content_CVPR_2019/papers/Haris_Recurrent_Back-Projection_Network_for_Video_Super-Resolution_CVPR_2019_paper.pdf)
   - Early published [arXiv](https://arxiv.org/abs/1903.10128)
   - Codes: [Pytorch](https://github.com/alterzero/RBPN-PyTorch)
   - Remark: **SOTA**, 7 frames, no explicit warping

- [FSTRN](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Fast_Spatio-Temporal_Residual_Network_for_Video_Super-Resolution_CVPR_2019_paper.pdf)
   - Early published [arXiv](https://arxiv.org/abs/1904.02870)
   - Codes: [Non-Official](https://github.com/summit1993/I-am-Super)
   - Remark: Modified 3D ResBlock

## Denoising
- CBDNet
   - Early published [arXiv](https://arxiv.org/abs/1807.04686)
   - Codes: [matlab](https://github.com/GuoShi28/CBDNet)
   - Remark: **SOTA**(nearly), blind

- [Noise2Void](http://openaccess.thecvf.com/content_CVPR_2019/papers/Krull_Noise2Void_-_Learning_Denoising_From_Single_Noisy_Images_CVPR_2019_paper.pdf)
   - Early published [arXiv](https://arxiv.org/abs/1811.10980)
   - Remark: A step further than Noise2Noise

- [FOCNet](http://openaccess.thecvf.com/content_CVPR_2019/papers/Jia_FOCNet_A_Fractional_Optimal_Control_Network_for_Image_Denoising_CVPR_2019_paper.pdf)
   - Codes: [empty](https://github.com/hsijiaxidian/FOCNet)
   - Remark: Ali DAMO, de-gaussian

- [Non-Local Meets Global An Integrated Paradigm for Hyperspectral Denoising](http://openaccess.thecvf.com/content_CVPR_2019/papers/He_Non-Local_Meets_Global_An_Integrated_Paradigm_for_Hyperspectral_Denoising_CVPR_2019_paper.pdf)
   - Remark: HSI denoising
   
- [Unprocessing Images for Learned Raw Denoising](http://openaccess.thecvf.com/content_CVPR_2019/papers/Brooks_Unprocessing_Images_for_Learned_Raw_Denoising_CVPR_2019_paper.pdf)
   - Early published [arXiv](https://arxiv.org/abs/1811.11127)
   - Remark: Google, RAW denoising

- [Model-blind Video Denoising Via Frame-to-frame Training](http://openaccess.thecvf.com/content_CVPR_2019/papers/Ehret_Model-Blind_Video_Denoising_via_Frame-To-Frame_Training_CVPR_2019_paper.pdf)
   - Early published [arXiv](https://arxiv.org/abs/1811.12766)
   - Remark: Video denoising

## Deblurring
- [Blind Image Deblurring with Local Maximum Gradient Prior](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Blind_Image_Deblurring_With_Local_Maximum_Gradient_Prior_CVPR_2019_paper.pdf)
- [Dynamic Scene Deblurring with Parameter Selective Sharing and Nested Skip Connections](http://openaccess.thecvf.com/content_CVPR_2019/papers/Gao_Dynamic_Scene_Deblurring_With_Parameter_Selective_Sharing_and_Nested_Skip_CVPR_2019_paper.pdf)
- [Deep Stacked Hierarchical Multi-patch Network for Image Deblurring](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Deep_Stacked_Hierarchical_Multi-Patch_Network_for_Image_Deblurring_CVPR_2019_paper.pdf)
- [Phase-only Image Based Kernel Estimation for Single Image Blind Deblurring](http://openaccess.thecvf.com/content_CVPR_2019/papers/Pan_Phase-Only_Image_Based_Kernel_Estimation_for_Single_Image_Blind_Deblurring_CVPR_2019_paper.pdf)
- [Recurrent Neural Networks with Intra-Frame Iterations for Video Deblurring](http://openaccess.thecvf.com/content_CVPR_2019/papers/Nah_Recurrent_Neural_Networks_With_Intra-Frame_Iterations_for_Video_Deblurring_CVPR_2019_paper.pdf)
- [A variational EM framework with adaptive edge selection for blind motion deblurring](http://openaccess.thecvf.com/content_CVPR_2019/papers/Yang_A_Variational_EM_Framework_With_Adaptive_Edge_Selection_for_Blind_CVPR_2019_paper.pdf)
- [Unsupervised Domain-Specific Deblurring via Disentangled Representations](http://openaccess.thecvf.com/content_CVPR_2019/papers/Lu_Unsupervised_Domain-Specific_Deblurring_via_Disentangled_Representations_CVPR_2019_paper.pdf)
- [DAVANet](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhou_DAVANet_Stereo_Deblurring_With_View_Aggregation_CVPR_2019_paper.pdf)

## Restoration
- [DuRN](http://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Dual_Residual_Networks_Leveraging_the_Potential_of_Paired_Operations_for_CVPR_2019_paper.pdf)
   - Early published [arXiv](https://arxiv.org/abs/1903.08817)
   - Remark: Dual residual block, dehaze/denoise/deblur

- [Attention-based Adaptive Selection of Operations for Image Restoration in the Presence of Unknown Combined Distortions](http://openaccess.thecvf.com/content_CVPR_2019/papers/Suganuma_Attention-Based_Adaptive_Selection_of_Operations_for_Image_Restoration_in_the_CVPR_2019_paper.pdf)
   - Early published [arXiv](https://arxiv.org/abs/1812.00733)
   - Codes: [Pytorch](https://github.com/sg-nm/Operation-wise-attention-network)
   - Remark: Attention, denoise

- [AdaFM](http://openaccess.thecvf.com/content_CVPR_2019/papers/He_Modulating_Image_Restoration_With_Continual_Levels_via_Adaptive_Feature_Modification_CVPR_2019_paper.pdf)
   - Early published [arXiv](https://arxiv.org/abs/1904.08118)
   - Codes: [Pytorch](https://github.com/hejingwenhejingwen/AdaFM)
   - Remark: SenseTime, oral

- [Heavy Rain Image Restoration: Integrating Physics Model and Conditional Adversarial Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Heavy_Rain_Image_Restoration_Integrating_Physics_Model_and_Conditional_Adversarial_CVPR_2019_paper.pdf)

## Todo
### Implement in VSR(Torch):
- [ ] IKC
- [ ] DPSR
- [ ] SAN
- [ ] FSTRN
- [ ] CBDNet
- [ ] AdaFM
