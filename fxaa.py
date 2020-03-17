import numpy as np

def getGrad(img):
    imgGrad0 = np.gradient(img, axis=0)
    imgGrad1 = np.gradient(img, axis=1)
    imgGradCut0 = np.absolute(imgGrad0)
    imgGradCut1 = np.absolute(imgGrad1)
    imgGradCut0 = imgGradCut0/np.amax(imgGradCut0)
    imgGradCut1 = imgGradCut1/np.amax(imgGradCut1)
    return imgGradCut0, imgGradCut1

def getMax(Zimg, Rimg, Gimg, Bimg, h, w):
    img = np.zeros((h, w, 1))
    for i in range(h):
        for j in range(w):
            img[h-i-1, j, :] = np.sqrt(np.sqrt(max(Zimg[h-i-1, j, :], Rimg[h-i-1, j, :], Gimg[h-i-1, j, :], Bimg[h-i-1, j, :])))
    return img


def onePass(img, Zimg, Rimg, Gimg, Bimg, h, w):


    ZimgGradCut0, ZimgGradCut1 = getGrad(Zimg)
    RimgGradCut0, RimgGradCut1 = getGrad(Rimg)
    GimgGradCut0, GimgGradCut1 = getGrad(Gimg)
    BimgGradCut0, BimgGradCut1 = getGrad(Bimg)

    imgGradCut0 = getMax(ZimgGradCut0, RimgGradCut0, GimgGradCut0, BimgGradCut0, h, w)
    imgGradCut1 = getMax(ZimgGradCut1, RimgGradCut1, GimgGradCut1, BimgGradCut1, h, w)


    def blur0(A,i,j):
        a11 = A[i-1,j-1,:]
        a12 = A[i,j-1,:]
        a13 = A[i+1,j-1,:]
        a21 = A[i-1,j,:]
        a22 = A[i,j,:]
        a23 = A[i+1,j,:]
        a31 = A[i-1,j+1,:]
        a32 = A[i,j+1,:]
        a33 = A[i+1,j+1,:]
        return 0.025*(a11 + a12 + a13 + a31 + a32 + a33) + 0.15*(a21 + a23) + 0.5*a22


    def blur1(A,i,j):
        a11 = A[i-1,j-1,:]
        a12 = A[i,j-1,:]
        a13 = A[i+1,j-1,:]
        a21 = A[i-1,j,:]
        a22 = A[i,j,:]
        a23 = A[i+1,j,:]
        a31 = A[i-1,j+1,:]
        a32 = A[i,j+1,:]
        a33 = A[i+1,j+1,:]
        return 0.025*(a11 + a21 + a31 + a13 + a23 + a33) + 0.15*(a12 + a32) + 0.5*a22

    def simpleBlur(A,i,j):
        a11 = A[i-1,j-1,:]
        a12 = A[i,j-1,:]
        a13 = A[i+1,j-1,:]
        a21 = A[i-1,j,:]
        a22 = A[i,j,:]
        a23 = A[i+1,j,:]
        a31 = A[i-1,j+1,:]
        a32 = A[i,j+1,:]
        a33 = A[i+1,j+1,:]
        return 0.1*(a11 + a21 + a31 + a13 + a23 + a33 + a12 + a32) + 0.2*a22


    def blur11(A,i,j):
        a11 = A[i-1,j-1,:]
        a13 = A[i+1,j-1,:]
        a31 = A[i-1,j+1,:]
        a33 = A[i+1,j+1,:]
        a20 = A[i-2,j,:]
        a21 = A[i-1,j,:]
        a22 = A[i,j,:]
        a23 = A[i+1,j,:]
        a24 = A[i+2,j,:]
        return 0.005*(a11+ a13 + a31 + a33) + 0.1*(a21 + a23) + 0.05*(a20 + a24) + 0.66*a22


    def blur00(A,i,j):
        a11 = A[i-1,j-1,:]
        a13 = A[i+1,j-1,:]
        a31 = A[i-1,j+1,:]
        a33 = A[i+1,j+1,:]
        a02 = A[i,j-2,:]
        a12 = A[i,j-1,:]
        a22 = A[i,j,:]
        a32 = A[i,j+1,:]
        a42 = A[i,j+2,:]
        return 0.005*(a11+ a13 + a31 + a33) + 0.1*(a12 + a32) + 0.05*(a02 + a42) + 0.66*a22


    imgAliasing = img.copy()
    print("Aliasing:")
    for i in range(2,h-2):
        print(100*i//h,"%")
        for j in range(2,w-2):
            if imgGradCut0[h-i-1][j] > 0:
                if imgGradCut1[h-i-1][j] > 0:
                    den = (imgGradCut0[h-i-1][j] + imgGradCut1[h-i-1][j])
                    imgAliasing[h-i-1, j, :] = imgGradCut0[h-i-1][j] * blur00(img,h-i-1,j)
                    imgAliasing[h-i-1, j, :] += imgGradCut1[h-i-1][j] * blur11(img,h-i-1,j)
                    imgAliasing[h-i-1, j, :] = imgAliasing[h-i-1, j, :] / den
                else:
                    imgAliasing[h-i-1, j, :] = imgGradCut0[h-i-1][j] * blur00(img,h-i-1,j)
            elif imgGradCut1[h-i-1][j] > 0:
                imgAliasing[h-i-1, j, :] = imgGradCut1[h-i-1][j] * blur11(img,h-i-1,j)
            imgAliasing[h-i-1, j, :] = np.clip(imgAliasing[h-i-1, j, :], 0, 1)
    return imgAliasing