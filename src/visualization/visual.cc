
#include "visual.h"


void DrawNone()
{
    pangolin::CreateWindowAndBind("Main", 640, 480);
    glEnable(GL_DEPTH_TEST);// 启动深度测试
    glEnable(GL_BLEND);// 启动颜色混合
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);// 颜色混合的方式

    pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 320, 0.2, 500),
                // 相机参数配置，高度，宽度，4个内参，最近/最远视距
                pangolin::ModelViewLookAt(2,0,2, 0,0,0, pangolin::AxisY)
                // 相机所在位置，相机所看点的位置，最后是相机轴方向
                );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));
    // 显示视图在窗口中的范围（下上左右），最后一个参数为视窗长宽比

    while(!pangolin::ShouldQuit())
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);// 清空颜色和深度缓存,刷新显示
        d_cam.Activate(s_cam);// 激活并设置状态矩阵
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);// 画背景

        pangolin::FinishFrame();// 最终显示
    }

}

