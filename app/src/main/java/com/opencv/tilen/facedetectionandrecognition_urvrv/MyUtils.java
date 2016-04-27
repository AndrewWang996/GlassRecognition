package com.opencv.tilen.facedetectionandrecognition_urvrv;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgproc;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.MatOfPoint;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

import java.util.List;
import java.util.ArrayList;

/**
 * Created by Tilen on 10.6.2015.
 */
public class MyUtils {


    private final static Scalar RED = new Scalar(255, 0, 0);
    /**
     * Written by Andy Wang
     * April 12 / 2016
     * @param bgrimg
     * @return List of 3 Objects:
     *      - contours: List<MatOfPoint>
     *      - minRect: List<RotatedRect>
     *      - minEllipse: List<RotatedRect>
     */
    public static List<Object> getCRE(Mat bgrimg) {
        Mat grayscale = new Mat();
        Imgproc.cvtColor(bgrimg, grayscale, Imgproc.COLOR_BGR2GRAY);

        Mat thresh_output = new Mat();
        Imgproc.threshold(grayscale, thresh_output, 0, 255, Imgproc.THRESH_BINARY);

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Point offset = new Point(0,0);
        Imgproc.findContours(thresh_output, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE, offset);

        int nContours = contours.size();
        List<RotatedRect> minRect = new ArrayList<>();
        List<RotatedRect> minEllipse = new ArrayList<>();

        for(int i=0; i<nContours; i++) {
            MatOfPoint cInt = contours.get(i);
            Point[] cArray = cInt.toArray();
            MatOfPoint2f c = new MatOfPoint2f( cArray );
            minRect.add(Imgproc.minAreaRect(c));
            if(cArray.length > 5) {
                minEllipse.add(Imgproc.fitEllipse(c));
            }
        }

        List<Object> combo = new ArrayList<>();
        combo.add(contours);
        combo.add(minRect);
        combo.add(minEllipse);
        return combo;
    }


    /**
     * Written by Andy Wang
     * April 26 2016
     *
     *
     */
    public static List<RotatedRect> computeRectangles(Mat bgrimg, List<RotatedRect> rectangles, int numIterations) {
        Mat imgCopy;
        List<RotatedRect> newRectangles = new ArrayList<>(rectangles);

        for(int i=0; i<numIterations; i++) {
            imgCopy = new Mat(bgrimg.size(), bgrimg.type());

            for (RotatedRect rect : newRectangles) {
                // Point[] box = new Point[4];
                // rect.points(box);
                Core.ellipse(imgCopy, rect, RED, -1);
            }

            List<Object> CRE = getCRE(imgCopy);
            newRectangles = (List<RotatedRect>) CRE.get(1);
        }

        return newRectangles;
    }

    /**
     * Written by Andy Wang
     * April 12 2016
     *
     * @param bgrimg
     * @param rectangles
     */
    public static void drawRectangles(Mat bgrimg, List<RotatedRect> rectangles) {
        for(RotatedRect rect : rectangles) {
            Core.ellipse(bgrimg, rect, RED, 2);
        }
    }

    public static RotatedRect getLargestRectangle(List<RotatedRect> rectangles) {
        double maxArea = -1;
        RotatedRect largestRect = null;
        for(RotatedRect rect : rectangles) {
            double area = rect.size.width * rect.size.height;
            if(area > maxArea) {
                maxArea = area;
                largestRect = rect.clone();
            }
        }
        return largestRect;
    }


    public static void addCaption(Mat img) {
        Core.putText(img, "Don't drink the coke!", new Point(50, 50), Core.FONT_HERSHEY_COMPLEX, 1.0, new Scalar(200, 200, 250), 1);
    }

    public static Mat captureRedRectangles(Mat img) {
        return captureRedRectangles(img, 40, 124, 11, true);
    }

    public static Mat captureRedRectangles(Mat img, int R, int BG, int KS, boolean onlyLargestRectangle) {
        Mat hsv = new Mat();
        Imgproc.cvtColor(img, hsv, Imgproc.COLOR_BGR2HSV);
        Mat mask = new Mat();
        Core.inRange(hsv, new Scalar(R, BG, BG), new Scalar(255, 255, 255), mask);
        Mat res = new Mat();
        Core.bitwise_and(img, img, res, mask);

        int kernel_size = KS;
        if( (kernel_size & 1) == 0) {
            kernel_size++;
        }

        Mat dst = new Mat();
        Imgproc.medianBlur(res, dst, kernel_size);

        img = dst;

        Mat drawingRects = dst.clone();
        List<Object> CRE = getCRE(drawingRects);
        List<RotatedRect> rectangles = (List<RotatedRect>)CRE.get(1);

        if(onlyLargestRectangle) {
            RotatedRect largestRect = getLargestRectangle(rectangles);
            rectangles = new ArrayList<>();
            if(largestRect != null) {
                rectangles.add(largestRect);
            }
        }

        rectangles = computeRectangles(img, rectangles, 5);
        drawRectangles(img, rectangles);
        addCaption(img);
        return img;
    }

    public static Bitmap matToBitmap(Mat inputPicture)
    {
        Mat convertedPicture = new Mat();
        Global.TestDebug("test : " +inputPicture.cols());
        Imgproc.cvtColor(inputPicture, convertedPicture, Imgproc.COLOR_RGB2BGRA);
        Bitmap bitmapPicture = Bitmap.createBitmap(inputPicture.cols(), inputPicture.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(convertedPicture, bitmapPicture);
        return bitmapPicture;
    }

    public static Mat bitmapToMat(Bitmap inputPicture)
    {
        Mat matPicture = new Mat();
        Utils.bitmapToMat(inputPicture, matPicture);
        Imgproc.cvtColor(matPicture, matPicture, Imgproc.COLOR_RGB2BGRA);
        return matPicture;
    }

    public static void saveBitmaps(Mat[] faceImages, Context mContext)
    {
        File cacheDir = mContext.getCacheDir();
        File file;
        FileOutputStream out;
        Bitmap bitmapPicture;
        for(int i = 0; i < faceImages.length;i++) {
            file = new File(cacheDir, "faceImage" + i);
            bitmapPicture = MyUtils.matToBitmap(faceImages[i]);
            try {
                out = new FileOutputStream(file);
                bitmapPicture.compress(
                        Bitmap.CompressFormat.PNG,
                        100, out);
                out.flush();
                out.close();

            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public static Bitmap[] loadBitmaps(int numberOfImages, Context mContext)
    {
        File cacheDir = mContext.getCacheDir();
        File file;
        FileInputStream fis;
        Bitmap[] faceImages = new Bitmap[numberOfImages];
        for(int i= 0; i < numberOfImages; i++) {
            file = new File(cacheDir, "faceImage" + i);
            fis = null;
            try {
                fis = new FileInputStream(file);
                faceImages[i] = BitmapFactory.decodeStream(fis);
                file.delete();
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
        }
        return faceImages;
    }

    //JavaCV library - testing purpose

    public static Bitmap IplImageToBitmap(opencv_core.IplImage source) {
        opencv_core.IplImage container = opencv_core.IplImage.create(source.width(), source.height(), opencv_core.IPL_DEPTH_8U, 4);
        opencv_imgproc.cvCvtColor(source, container, opencv_imgproc.CV_BGR2RGBA);
        Bitmap bitmap = Bitmap.createBitmap(source.width(), source.height(), Bitmap.Config.ARGB_8888);
        bitmap.copyPixelsFromBuffer(container.createBuffer());
        return bitmap;
    }

    public static opencv_core.IplImage BitmapToIplImage(Bitmap source) {
        opencv_core.IplImage container = opencv_core.IplImage.create(source.getWidth(), source.getHeight(), opencv_core.IPL_DEPTH_8U, 4);

        source.copyPixelsToBuffer(container.createBuffer());
        opencv_imgproc.cvCvtColor(container, container, opencv_imgproc.CV_BGR2RGBA); // works now for javacv library
        return container;
    }
}
