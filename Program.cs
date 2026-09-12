using Image_Enhance_App_Backend_API;
using Microsoft.AspNetCore.Mvc;

var builder = WebApplication.CreateBuilder(args);

// Cấu hình Kestrel lắng nghe cổng 5123
builder.WebHost.ConfigureKestrel(serverOptions =>
{
    serverOptions.ListenAnyIP(5123);
});

// Đăng ký Service Singleton
builder.Services.AddSingleton<IFaceEnhancerService, FaceEnhancerService>();

var app = builder.Build();

app.MapGet("/", () => Results.Ok("Backend C# AI Server đang chạy!"));

// Endpoint nhận ảnh và xử lý AI
app.MapPost("/enhance", async (IFormFile file, [FromServices] IFaceEnhancerService enhancerService) =>
{
    if (file == null || file.Length == 0)
        return Results.BadRequest("Chưa nhận được file ảnh!");

    try
    {
        using var stream = file.OpenReadStream();
        byte[] resultBytes = await enhancerService.EnhanceFaceAsync(stream);
        return Results.File(resultBytes, "image/jpeg");
    }
    catch (Exception ex)
    {
        return Results.Problem($"Lỗi Backend: {ex.Message}");
    }
}).DisableAntiforgery();

app.Run();