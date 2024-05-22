package com.example.zanpan;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.graphics.RectF;
import android.graphics.Region;
import android.util.AttributeSet;
import android.view.GestureDetector;
import android.view.MotionEvent;
import android.view.ScaleGestureDetector;
import android.view.View;
import android.widget.ImageView;

import java.util.ArrayList;
import java.util.List;

public class DrawingView extends View {
    private List<DrawLine> lines;
    private Paint paint;
    private Path path;
    private Path eraserpath;
    private boolean isEraserMode = false;

    private boolean isDrawMode = true;



    class DrawLine {
        private Paint paint;
        private Path path;

        DrawLine(Path path, Paint paint) {
            this.paint = new Paint(paint);
            this.path = new Path(path);
        }



        void draw(Canvas canvas, Path path) {
            canvas.drawPath(this.path, this.paint);
            if(isEraserMode){
                this.paint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.CLEAR));
                canvas.drawPath(path, this.paint);
                this.paint.setXfermode(null);
            }

        }

        boolean contains(float x, float y) {
            RectF bounds = new RectF();
            path.computeBounds(bounds, true);
            Region region = new Region();
            region.setPath(path, new Region((int) bounds.left, (int) bounds.top, (int) bounds.right, (int) bounds.bottom));
            return region.contains((int) x, (int) y);
        }
    }

    public DrawingView(Context context) {
        super(context);
    }

    public DrawingView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
        path = new Path();
        paint = new Paint();
//        paint.setStyle(Paint.Style.STROKE);
//        paint.setAntiAlias(true);
//        paint.setStrokeWidth(10);
        paint.setColor(Color.RED);
        lines = new ArrayList<>();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        for (DrawLine line : lines) {
            line.draw(canvas, path);
        }
        canvas.drawPath(path, paint);
    }

public boolean getDrawMode(){
        return this.isDrawMode;
    }

    public void setDrawMode(boolean flag){
        this.isDrawMode = flag;
    }



    @Override
    public boolean onTouchEvent (MotionEvent event){
        if(isDrawMode) {
            float x = event.getX();
            float y = event.getY();

            switch (event.getAction()) {
                case MotionEvent.ACTION_DOWN:
                    if (isEraserMode) {
                        removePathAt(x, y);
                    }
                    path.moveTo(x, y);

                    break;
                case MotionEvent.ACTION_MOVE:

                    path.lineTo(x, y);

                    break;
                case MotionEvent.ACTION_UP:

                    path.lineTo(x, y);
                    lines.add(new DrawLine(path, paint));
                    path.reset();
                    break;
            }
            invalidate();
            return true;
        }else{
            return false;
        }
}

    private void removePathAt(float x, float y) {
        for (int i = lines.size() - 1; i >= 0; i--) {
            DrawLine line = lines.get(i);
            if (line.contains(x, y)) {
                lines.remove(i);
                break;
            }
        }
    }
    public void setPenColor(int color) {
        paint.setColor(color);
        isEraserMode = false;
    }

    public void setPenWidth(int size) {
        paint.setStrokeWidth(size);
    }

    public void setEraserMode() {
        isEraserMode = true;
        invalidate();
    }

    public void clearDrawing() {
        lines.clear();
        path.reset();
        invalidate();
    }

}



