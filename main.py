import cv2
import streamlit as st
from PIL import Image
import numpy as np

def LapSRN_x2(img):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = 'models/LapSRN/LapSRN_x2.pb'
    sr.readModel(path)
    sr.setModel('lapsrn',2)
    img = np.array(img.convert("RGB"))
    result = sr.upsample(img)
    st.image(result, caption='LapSRN_x2')
    st.write('Shape of SR image is: ',np.shape(result))


def ESPCN_x2(img):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = 'models/ESPCN/ESPCN_x2.pb'
    sr.readModel(path)
    sr.setModel('espcn',2)
    img = np.array(img.convert("RGB"))
    result = sr.upsample(img)
    st.image(result, caption='ESPCN_x2')
    st.write('Shape of SR image is: ',np.shape(result))

def FSRCNN_x2(img):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = 'models/FSRCNN/FSRCNN_x2.pb'
    sr.readModel(path)
    sr.setModel('fsrcnn',2)
    img = np.array(img.convert("RGB"))
    result = sr.upsample(img)
    st.image(result, caption='FSRCNN_x2')
    st.write('Shape of SR image is: ',np.shape(result))

def EDSR_x2(img):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = 'models/EDSR/EDSR_x2.pb'
    sr.readModel(path)
    sr.setModel('edsr',2)
    img = np.array(img.convert("RGB"))
    result = sr.upsample(img)
    st.image(result, caption='EDSR_x2')
    st.write('Shape of SR image is: ',np.shape(result))
# ////////////////////////////////////////////////////////


def ESPCN_x3(img):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = 'models/ESPCN/ESPCN_x3.pb'
    sr.readModel(path)
    sr.setModel('espcn',3)
    img = np.array(img.convert("RGB"))
    result = sr.upsample(img)
    st.image(result, caption='ESPCN_x3')
    st.write('Shape of SR image is: ',np.shape(result))

def FSRCNN_x3(img):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = 'models/FSRCNN/FSRCNN_x3.pb'
    sr.readModel(path)
    sr.setModel('fsrcnn',3)
    img = np.array(img.convert("RGB"))
    result = sr.upsample(img)
    st.image(result, caption='FSRCNN_x3')
    st.write('Shape of SR image is: ',np.shape(result))

def EDSR_x3(img):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = 'models/EDSR/EDSR_x3.pb'
    sr.readModel(path)
    sr.setModel('edsr',3)
    img = np.array(img.convert("RGB"))
    result = sr.upsample(img)
    st.image(result, caption='EDSR_x3')
    st.write('Shape of SR image is: ',np.shape(result))
# ///////////////////////////////////////////////////////
def LapSRN_x4(img):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = 'models/LapSRN/LapSRN_x4.pb'
    sr.readModel(path)
    sr.setModel('lapsrn',4)
    img = np.array(img.convert("RGB"))
    result = sr.upsample(img)
    st.image(result, caption='LapSRN_x4')
    st.write('Shape of SR image is: ',np.shape(result))


def ESPCN_x4(img):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = 'models/ESPCN/ESPCN_x4.pb'
    sr.readModel(path)
    sr.setModel('espcn',4)
    img = np.array(img.convert("RGB"))
    result = sr.upsample(img)
    st.image(result, caption='ESPCN_x4')
    st.write('Shape of SR image is: ',np.shape(result))

def FSRCNN_x4(img):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = 'models/FSRCNN/FSRCNN_x4.pb'
    sr.readModel(path)
    sr.setModel('fsrcnn',4)
    img = np.array(img.convert("RGB"))
    result = sr.upsample(img)
    st.image(result, caption='FSRCNN_x4')
    st.write('Shape of SR image is: ',np.shape(result))

def EDSR_x4(img):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = 'models/EDSR/EDSR_x4.pb'
    sr.readModel(path)
    sr.setModel('edsr',4)
    img = np.array(img.convert("RGB"))
    result = sr.upsample(img)
    st.image(result, caption='EDSR_x4')
    st.write('Shape of SR image is: ',np.shape(result))

# //////////////////////////////////////////////////

def LapSRN_x8(img):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = 'models/LapSRN/LapSRN_x8.pb'
    sr.readModel(path)
    sr.setModel('lapsrn',8)
    img = np.array(img.convert("RGB"))
    result = sr.upsample(img)
    st.image(result, caption='LapSRN_x8')
    st.write('Shape of SR image is: ',np.shape(result))



def main():

    st.title('Super Resolution with pre-trained Deep learning algorithms')

    choice = st.sidebar.selectbox('Select zoom',('2X', '3X', '4X', '8X'))


    user_img = st.file_uploader("Upload your image", type=['png', 'jpg', 'jpeg'])

    if user_img ==  None:
        st.echo('Please upload your image')
    elif user_img is not None:
        img = Image.open(user_img)
        st.image(img, caption='User Image')
        st.write('Shape of user image is: ',np.shape(img))

        if choice == '2X':
            options = st.selectbox('Choose SR algorithm', ('EDSR_x2','ESPCN_x2','FSRCNN_x2','LapSRN_x2'))
            if st.button('Process'):
                if options == 'EDSR_x2':
                    EDSR_x2(img)

                elif options == 'ESPCN_x2':
                    ESPCN_x2(img)

                elif options == 'FSRCNN_x2':
                    FSRCNN_x2(img)

                elif options == 'LapSRN_x2':
                    LapSRN_x2(img)


        elif choice == '3X':
            options = st.selectbox('Choose SR algorithm', ('EDSR_x3','ESPCN_x3','FSRCNN_x3'))
            if st.button('Process'):
                if options == 'EDSR_x3':
                    EDSR_x3(img)

                elif options == 'ESPCN_x3':
                    ESPCN_x3(img)

                elif options == 'FSRCNN_x3':
                    FSRCNN_x3(img)

        elif choice == '4X':

            options = st.selectbox('Choose SR algorithm', ('EDSR_x4','ESPCN_x4','FSRCNN_x4','LapSRN_x4'))
            if st.button('Process'):
                if options == 'EDSR_x4':
                    EDSR_x4(img)

                elif options == 'ESPCN_x4':
                    ESPCN_x4(img)

                elif options == 'FSRCNN_x4':
                    FSRCNN_x4(img)

                elif options == 'LapSRN_x4':
                    LapSRN_x4(img)

        elif choice == '8X':
            if st.button('Process LapSRN_x8'):
                LapSRN_x8(img)

    

if __name__ == '__main__':
    main()