package com.opencv.tilen.facedetectionandrecognition_urvrv;

import android.content.Context;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.objdetect.CascadeClassifier;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

/**
 * Created by Andy on 1.30.2016.
 */
public class CokeDetection {

    // lbp classifier
    private final int lbpFrontalFaceId = R.raw.lbpcascade_frontalface;
    private String lbpFrontalFaceClassifier;
    private String lbpFrontalFaceClassifierPath;
    private final String lpbFrontalFaceClassifierFilename = "lbp_cascade_frontalface";

    // haar classifier
    private final int haarFrontalFaceId = R.raw.haarcascade_frontalface_default;
    private String haarfrontalFaceClassifier;
    private String haarfrontalFaceClassifierPath;
    private final String haarFrontalFaceClassifierFilename = "haar_cascade_frontalface";

    // haar classifier for coke bottles
    private final int haarCokeId = R.raw.haarcascade_coke;
    private String haarCokeClassifier;
    private String haarCokeClassifierPath;
    private final String haarCokeClassifierFilename = "haarcascade_coke";


    private Context mContext;
    private CascadeClassifier cokeDetectorCascadeClassifier;
    private static CokeDetection cokeDetector = null;
    private int numberOfCokesInCurrentImage;

    private String lastClassifierPath; // for optimization


    public String getHaarCokeClassifierPath() {
        return haarCokeClassifierPath;
    }


    public int getNumberOfCokesInCurrentImage() {
        return numberOfCokesInCurrentImage;
    }

    public static CokeDetection getInstance(Context context) {
        //singleton
        if (cokeDetector == null) {
            cokeDetector = new CokeDetection(context);
        }
        return cokeDetector;
    }

    private CokeDetection(Context context) {
        lastClassifierPath = "";
        mContext = context;

        /*
        // lbp classifier
        lbpFrontalFaceClassifierPath = getClassifierFilename(lbpFrontalFaceId, lpbFrontalFaceClassifierFilename);
        lbpFrontalFaceClassifier = loadClassifierString(lbpFrontalFaceId, lpbFrontalFaceClassifierFilename);

        // haar classifier
        haarfrontalFaceClassifierPath = getClassifierFilename(haarFrontalFaceId, haarFrontalFaceClassifierFilename);
        haarfrontalFaceClassifier = loadClassifierString(haarFrontalFaceId, haarFrontalFaceClassifierFilename);
        setUpCascadeClassifier(lbpFrontalFaceClassifierPath);
        */

        // haar coke classifier
        haarCokeClassifierPath = getClassifierFilename(haarCokeId, haarCokeClassifierFilename);
        haarCokeClassifier = loadClassifierString(haarCokeId, haarCokeClassifierFilename);
        setUpCascadeClassifier(haarCokeClassifierPath);
    }

    private String loadClassifierString(int resourceId, String classifierName) {
        StringBuilder stringBuilder = new StringBuilder();
        try {
            InputStream inputStream = mContext.getResources().openRawResource(resourceId);

            BufferedReader input = new BufferedReader(new InputStreamReader(inputStream), 1024 * 8);
            try {
                String line = null;
                while ((line = input.readLine()) != null) {
                    stringBuilder.append(line);
                    stringBuilder.append("\n");
                }
            } finally {
                input.close();
            }
        } catch (FileNotFoundException ex) {
            Global.ErrorDebug("CokeDetection.loadClassifierString(): Couldn't find the file "
                    + resourceId + " " + ex);
            return "";
        } catch (IOException ex) {
            Global.ErrorDebug("CokeDetection.loadClassifierString(): Error reading file "
                    + resourceId + " " + ex);
            return "";
        }
        String classifierString = stringBuilder.toString();
        Global.InfoDebug("CokeDetection.loadClassifierString(): " + classifierName + " text: " + classifierString);
        return classifierString;
    }

    // we need to copy it from resources to temporary directory
    private String getClassifierFilename(int resourceId, String classifierName) {
        try {
            InputStream inputStream = mContext.getResources().openRawResource(resourceId);
            // File cascadeDirectory = mContext.getDir("cascade", Context.MODE_PRIVATE);
            //File cascadeFile = new File(cascadeDirectory, "lbpcascade_frontalface.xml");
            File cascadeFile = File.createTempFile(classifierName, ".xml");
            FileOutputStream outputStream = new FileOutputStream(cascadeFile);

            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, bytesRead);
            }
            inputStream.close();
            outputStream.close();


            String classifierPath = cascadeFile.getAbsolutePath();
            Global.LogDebug("CokeDetection.getClassifierName(): Path of file (" + classifierName + "): " + classifierPath);
            return classifierPath;

        } catch (Exception e) {
            Global.ErrorDebug("CokeDetection.getClassifierName(): Error loading file from raw to temporary location: " + e.toString());
            return "";
        }


        /* doesn't work
        String path = "android.resource://"+ mContext.getPackageName() + "/raw/lbpcascade_frontalface.xml";// + R.raw.lbpcascade_frontalface;
        Uri uri = Uri.parse(path);
        lbpFrontalFaceClassifierPath = uri.toString();
        */
    }

    public void setUpCascadeClassifier(String classifierPath) {
        if(!classifierPath.equals(lastClassifierPath)) { // if current classifier is not the same as chosen
            cokeDetectorCascadeClassifier = new CascadeClassifier(classifierPath);
            lastClassifierPath = classifierPath;
            if (cokeDetectorCascadeClassifier.empty() == true)
                Global.ErrorDebug("CokeDetection.getFaceDetectionPicture(): Classifier has not been loaded. ClassifierFilePath: " + lbpFrontalFaceClassifierPath);
        }
    }

    /**
     * picture with rectangles for all cokes
     **/
    public Mat getCokeDetectionPicture(Mat inputPicture) {
        //inputPicture = localPictures.getlocalPicture();
        //Global.TestDebug("CokeDetection.getCokeDetectionPicture: inputPicture " + inputPicture.cols());

        // MatOfRect is a special container class for Rect. Probably such as vector in c++ (MatOf...)
        MatOfRect faceDetectionRectangles = new MatOfRect();
        inputPicture = inputPicture.clone();// necessary a clone to avoid referencing the same object
        cokeDetectorCascadeClassifier.detectMultiScale(inputPicture, faceDetectionRectangles);
        Rect[] rectangles = faceDetectionRectangles.toArray();
        Global.LogDebug("CokeDetection.getCokeDetectionPicture() Number of faces: " + rectangles.length);
        numberOfCokesInCurrentImage = rectangles.length;
        for (Rect rect : rectangles) {
            Core.rectangle(inputPicture, rect.tl(), rect.br(), new Scalar(154, 250, 0));
        }
        return inputPicture;
    }

    /**
     * get pictures of all faces in main picture
     **/
    public Mat[] getCokePictures(Mat inputPicture) {
        MatOfRect cokeDetectionRectangles = new MatOfRect();
        cokeDetectorCascadeClassifier.detectMultiScale(inputPicture, cokeDetectionRectangles);
        Rect[] rectangles = cokeDetectionRectangles.toArray();
        if (rectangles.length == 0) // not coke detected
            return null;
        Mat[] cokePictures = new Mat[rectangles.length];
        for (int i = 0; i < rectangles.length; i++)
            cokePictures[i] = new Mat(inputPicture, rectangles[i]);
        //!if there are some error about wrong picture or destroyed clone image!
        return cokePictures;
    }

}
