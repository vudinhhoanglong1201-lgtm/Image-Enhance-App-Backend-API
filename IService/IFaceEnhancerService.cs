namespace Image_Enhance_App_Backend_API;

public interface IFaceEnhancerService
{
    Task<byte[]> EnhanceFaceAsync(Stream inputStream);
}