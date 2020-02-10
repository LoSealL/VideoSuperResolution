# Collections Specially for ICCV 2019
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

## VSR
- [RBPN](http://openaccess.thecvf.com/content_CVPR_2019/papers/Haris_Recurrent_Back-Projection_Network_for_Video_Super-Resolution_CVPR_2019_paper.pdf)
   - Early published [arXiv](https://arxiv.org/abs/1903.10128)
   - Codes: [Pytorch](https://github.com/alterzero/RBPN-PyTorch)
   - Remark: **SOTA**, 7 frames, no explicit warping

## Denoising
- [Noise2Void](http://openaccess.thecvf.com/content_CVPR_2019/papers/Krull_Noise2Void_-_Learning_Denoising_From_Single_Noisy_Images_CVPR_2019_paper.pdf)
   - Early published [arXiv](https://arxiv.org/abs/1811.10980)
   - Remark: A step further than Noise2Noise

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

## Todo
### Implement in VSR:
- [ ] IKC
