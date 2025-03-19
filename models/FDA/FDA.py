import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

class FDA:
    def __init__(self):
        pass
    
    def convert_source_to_target_pil(self, src_img, trg_img, L=0.01):
        # exchange magnitude
        # input: src_img, trg_img

        src_img_np = src_img #.cpu().numpy()
        trg_img_np = trg_img #.cpu().numpy()

        # get fft of both source and target
        fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )
        fft_trg_np = np.fft.fft2( trg_img_np, axes=(-2, -1) )

        # extract amplitude and phase of both ffts
        amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
        amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

        # mutate the amplitude part of source with target
        amp_src_ = self.low_freq_mutate_np( amp_src, amp_trg, L=L )

        # mutated fft of source
        fft_src_ = amp_src_ * np.exp( 1j * pha_src )

        # get the mutated image
        src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
        src_in_trg = np.real(src_in_trg)
        
        # Convert back to (H, W, C) format
        src_in_trg = src_in_trg.transpose((1, 2, 0))
        
        # Normalize the image data to 0-1 range
        src_in_trg = np.clip(src_in_trg / 255.0, 0, 1)

        return Image.fromarray((src_in_trg * 255).astype(np.uint8)) 
       
    def low_freq_mutate_np(self, amp_src, amp_trg, L=0.1 ):
        a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
        a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )

        _, h, w = a_src.shape
        b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
        c_h = np.floor(h/2.0).astype(int)
        c_w = np.floor(w/2.0).astype(int)

        h1 = c_h-b
        h2 = c_h+b+1
        w1 = c_w-b
        w2 = c_w+b+1

        a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
        a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
        return a_src