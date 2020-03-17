############################################
import numpy as np
import cv2

def get_BW(m_gray):
    # cv2.imshow("adfs", m_gray)
    # cv2.waitKey()
    _, BW = cv2.threshold(m_gray, 225, 255, cv2.THRESH_BINARY_INV)
    # BW = cv2.adaptiveThreshold(m_RGB, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 10)
    # cv2.imshow("adfs", BW)
    # cv2.waitKey()

    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((5, 5), np.uint8)
    BW = cv2.dilate(BW, kernel, iterations=1)
    BW = cv2.erode(BW, kernel, iterations=1)

    # Copy the thresholded image.
    im_floodfill = BW.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = BW.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255);

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = BW | im_floodfill_inv
    # cv2.imshow("Foreground", im_out)
    # cv2.waitKey(0)
    #
    return im_out

############################################
def main(fileName):
    label = ["Bear","fist","Cox","Lip","Potato","Timer","Car","Boy","Flash","First finger","Flower","Air Plane","Buddha","Kiding","Music","Boyes","Face","Congrat","Horse","Sleep","Angrey","Flame","Boy2","Finger","Big smile","smile2","smile","sadness","Happy","Apple"]
    res_dictionary = {}
    n=-1
    for x in range(97, 97 + 26):
        n+=1
        res_dictionary[str(x)]=n

    for x in range(49,49+5):
        n += 1
        res_dictionary[str(x)]=n

    train_flag = 0
    if train_flag == 0:
        samples = np.loadtxt("samples").astype(np.float32)
        response = np.loadtxt("response").astype(np.float32)
        knn_model = cv2.ml.KNearest_create()
        m_k = len(samples);
        knn_model.train(samples,cv2.ml.ROW_SAMPLE,response)

    #-----------------------------------------
    m_RGB = cv2.imread(fileName)
    m_gray = cv2.imread(fileName, 0)
    cv2.imshow("RGB image", m_RGB)
    cv2.waitKey()
    #-----------------------------------------
    im_out = get_BW(m_gray)
    _, contours, _= cv2.findContours(im_out, 1, 2)

    # -----------------------------------------
    sample_width = 28; sample_height = 28;

    # -----------------------------------------
    samples = [];    response = []

    for cnt in contours:
        if cv2.contourArea(cnt)<300: continue;
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(m_RGB, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # -----------------------------------------
        m_ROI = np.copy(m_gray[y:y + h, x:x + w])
        m_ROI = cv2.resize(m_ROI, dsize=(sample_width, sample_height))
        sample = np.reshape(m_ROI, (1, sample_width * sample_height))
        cv2.imshow("Foreground", m_RGB)
        k = cv2.waitKey(0)
        cv2.destroyAllWindows()
        # -----------------------------------------
        if train_flag == 1:
            cv2.imshow("Foreground", m_RGB)
            cv2.destroyAllWindows()


            if k == 27:  # Esc key to stop
                break
            else:
                samples.append(list(sample[0]))
                response.append(np.float(k))
        else:
            sample = sample.astype(np.float32)
            ret, results, neighbours, dist = knn_model.findNearest(sample,m_k)
            m_predict_label = label[res_dictionary[str(np.int16(neighbours[0][0]))]]
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(m_RGB, m_predict_label, (x,y), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow("Foreground", m_RGB)
    cv2.waitKey(0)

    if train_flag == 1:
        np.savetxt("samples",samples)
        np.savetxt("response", response)

############################################
if __name__ == "__main__":
    main('image.png')

