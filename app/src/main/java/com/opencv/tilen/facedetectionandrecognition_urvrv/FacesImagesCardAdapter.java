package com.opencv.tilen.facedetectionandrecognition_urvrv;

import android.content.Context;
import android.graphics.Bitmap;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ImageView;

import com.google.android.glass.widget.CardBuilder;
import com.google.android.glass.widget.CardScrollAdapter;

import org.opencv.core.Mat;

/**
 * Created by Tilen on 17.6.2015.
 */
public class FacesImagesCardAdapter extends CardScrollAdapter{
    private final Context mContext;
    private final String imageName;
    private final Mat[] faceImages;

    public FacesImagesCardAdapter(Context mContext, String imageName, Mat[] faceImages) {
        this.mContext = mContext;
        this.imageName = imageName;
        this.faceImages = faceImages;
    }

    @Override
    public int getCount() {
        return faceImages.length;
    }

    @Override
    public Object getItem(int position) {
        return faceImages[position];
    }

    @Override
    public View getView(int position, View convertView, ViewGroup parent) {
        Bitmap bitmapImage = LocalPicturesDetection.matToBitmap(faceImages[position]);
        CardBuilder card = new CardBuilder(mContext, CardBuilder.Layout.EMBED_INSIDE)
                .setEmbeddedLayout(R.layout.activity_static_images)
                .setFootnote(imageName + " " + (position + 1)); // 0 - picture is first picture, ...
        View view = card.getView(convertView, parent);
        ImageView ivPicture = (ImageView) view.findViewById(R.id.ivPicture);
        ivPicture.setImageBitmap(bitmapImage);
        return view;
    }

    @Override
    public int getPosition(Object item) {
        for (int i = 0; i < faceImages.length; i++) {
            if (getItem(i).equals(item)) {
                return i;
            }
        }
        return AdapterView.INVALID_POSITION;

    }
}