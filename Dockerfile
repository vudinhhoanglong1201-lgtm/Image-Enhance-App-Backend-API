# Stage 1: Build project với SDK .NET 10
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src

# Copy file project .csproj và khôi phục các gói NuGet
COPY ["Image Enhance App Backend API.csproj", "./"]
RUN dotnet restore "Image Enhance App Backend API.csproj"

# Copy toàn bộ mã nguồn và biên dịch Release
COPY . .
RUN dotnet publish "Image Enhance App Backend API.csproj" -c Release -o /app/publish

# Stage 2: Môi trường Runtime .NET 10
FROM mcr.microsoft.com/dotnet/aspnet:10.0 AS final
WORKDIR /app
COPY --from=build /app/publish .

# Render cấp port mặc định cho container
ENV ASPNETCORE_URLS=http://+:8080
EXPOSE 8080

ENTRYPOINT ["dotnet", "Image Enhance App Backend API.dll"]