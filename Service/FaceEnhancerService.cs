using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

public interface IFaceEnhancerService
{
    Task<byte[]> EnhanceFaceAsync(Stream inputStream);
}

public class FaceEnhancerService : IFaceEnhancerService
{
    private readonly InferenceSession? _sessionRealEsrgan;

    public FaceEnhancerService()
    {
        string modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "realesrgan-x4.onnx");
        if (File.Exists(modelPath))
        {
            var options = new Microsoft.ML.OnnxRuntime.SessionOptions
            {
                EnableCpuMemArena = true,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_BASIC
            };
            _sessionRealEsrgan = new InferenceSession(modelPath, options);
        }
    }

    public async Task<byte[]> EnhanceFaceAsync(Stream inputStream)
    {
        if (_sessionRealEsrgan == null)
            throw new FileNotFoundException("Mô hình Real-ESRGAN chưa được nạp!");

        if (inputStream.CanSeek) inputStream.Position = 0;

        // 1. Tải ảnh đầu vào
        using var src = await Image.LoadAsync<Rgb24>(inputStream);
        int origW = src.Width;
        int origH = src.Height;

        // Các thông số Tiling tương tự Form1
        int inputModelSize = 64;
        int pad = 8;
        int coreStep = inputModelSize - (pad * 2); // 48px
        int scale = 4;

        // 2. Tạo ảnh đầu ra rỗng kích thước gấp 4 lần
        using var outputImage = new Image<Rgb24>(origW * scale, origH * scale);

        string inputName = _sessionRealEsrgan.InputMetadata.Keys.First();
        var inputTensor = new DenseTensor<float>(new[] { 1, 3, 64, 64 });

        // 3. Vòng lặp cắt và ghép từng ô
        for (int y = 0; y < origH; y += coreStep)
        {
            for (int x = 0; x < origW; x += coreStep)
            {
                int cropX = Math.Max(0, x - pad);
                int cropY = Math.Max(0, y - pad);
                int cropW = Math.Min(origW - cropX, inputModelSize);
                int cropH = Math.Min(origH - cropY, inputModelSize);

                // Cắt tile gốc từ ảnh
                using var tile = src.Clone(ctx => ctx.Crop(new Rectangle(cropX, cropY, cropW, cropH)));

                // Nếu tile nhỏ hơn 64x64 thì pad bằng màu lề
                using var paddedTile = new Image<Rgb24>(inputModelSize, inputModelSize);
                paddedTile.Mutate(ctx => ctx.DrawImage(tile, new Point(0, 0), 1f));

                // Đưa dữ liệu ảnh vào Tensor [1, 3, 64, 64]
                paddedTile.ProcessPixelRows(accessor =>
                {
                    for (int ty = 0; ty < inputModelSize; ty++)
                    {
                        var row = accessor.GetRowSpan(ty);
                        for (int tx = 0; tx < inputModelSize; tx++)
                        {
                            inputTensor[0, 0, ty, tx] = row[tx].R / 255.0f;
                            inputTensor[0, 1, ty, tx] = row[tx].G / 255.0f;
                            inputTensor[0, 2, ty, tx] = row[tx].B / 255.0f;
                        }
                    }
                });

                // Chạy ONNX Inference
                var inputs = new[] { NamedOnnxValue.CreateFromTensor(inputName, inputTensor) };
                using var results = _sessionRealEsrgan.Run(inputs);
                var outputTensor = results.First().AsTensor<float>();

                // Tạo tile kết quả 256x256
                using var outTileMat = new Image<Rgb24>(256, 256);
                outTileMat.ProcessPixelRows(accessor =>
                {
                    for (int ty = 0; ty < 256; ty++)
                    {
                        var row = accessor.GetRowSpan(ty);
                        for (int tx = 0; tx < 256; tx++)
                        {
                            byte r = (byte)Math.Clamp(outputTensor[0, 0, ty, tx] * 255.0f, 0, 255);
                            byte g = (byte)Math.Clamp(outputTensor[0, 1, ty, tx] * 255.0f, 0, 255);
                            byte b = (byte)Math.Clamp(outputTensor[0, 2, ty, tx] * 255.0f, 0, 255);
                            row[tx] = new Rgb24(r, g, b);
                        }
                    }
                });

                // Tính toán vùng lõi chính xác loại bỏ phần lề thừa
                int offsetXInTile = (x - cropX) * scale;
                int offsetYInTile = (y - cropY) * scale;
                int validCoreW = Math.Min(origW - x, coreStep) * scale;
                int validCoreH = Math.Min(origH - y, coreStep) * scale;

                using var coreOutput = outTileMat.Clone(ctx => ctx.Crop(new Rectangle(offsetXInTile, offsetYInTile, validCoreW, validCoreH)));

                // Vẽ ô vừa xử lý vào ảnh tổng
                outputImage.Mutate(ctx => ctx.DrawImage(coreOutput, new Point(x * scale, y * scale), 1f));
            }
        }

        // 4. Xuất mảng Byte PNG/JPEG
        using var ms = new MemoryStream();
        await outputImage.SaveAsJpegAsync(ms);
        return ms.ToArray();
    }
}