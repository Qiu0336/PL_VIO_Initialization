
#include "camera/CameraFactory.h"

namespace camodocal
{

boost::shared_ptr<CameraFactory> CameraFactory::m_instance;

CameraFactory::CameraFactory()
{
}

boost::shared_ptr<CameraFactory>
CameraFactory::instance(void)
{
    if (m_instance.get() == 0)
    {
        m_instance.reset(new CameraFactory);
    }

    return m_instance;
}

CameraPtr CameraFactory::generateCamera(Camera::ModelType modelType,
                              const std::string& cameraName,
                              cv::Size imageSize) const
{
    PinholeCameraPtr camera(new PinholeCamera);

    PinholeCamera::Parameters params = camera->getParameters();
    params.cameraName() = cameraName;
    params.imageWidth() = imageSize.width;
    params.imageHeight() = imageSize.height;
    camera->setParameters(params);
    return camera;
}

CameraPtr CameraFactory::generateCameraFromYamlFile(const std::string& filename)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);

    if (!fs.isOpened())
    {
        return CameraPtr();
    }

    PinholeCameraPtr camera(new PinholeCamera);

    PinholeCamera::Parameters params = camera->getParameters();
    params.readFromYamlFile(filename);
    camera->setParameters(params);
    return camera;

    return CameraPtr();
}

}
