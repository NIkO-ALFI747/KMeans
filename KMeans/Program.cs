using KMeans.Services;
using KMeans.Services.Interfaces;
using KMeans.Services.Interfaces.Internal;
using KMeans.Services.Interfaces.Internal.IKMeansAlgorithm;
using KMeans.Services.Interfaces.Internal.IOptimalKFinder;
using KMeans.Services.Internal;
using KMeans.Services.Internal.KMeansAlgorithm;
using KMeans.Services.Internal.OptimalKFinder;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddScoped<IKMeansAlgorithm, KMeansAlgorithm>();
builder.Services.AddScoped<IKMeansAlgorithmAggregator, KMeansAlgorithmAggregator>();
builder.Services.AddScoped<IKMeansAssignmentProcessor, KMeansAssignmentProcessor>();
builder.Services.AddScoped<IKMeansCentroidUpdater, KMeansCentroidUpdater>();
builder.Services.AddScoped<IKMeansInitializer, KMeansInitializer>();

builder.Services.AddScoped<IElbowMethodCalculator, ElbowMethodCalculator>();
builder.Services.AddScoped<IOptimalKFinder, OptimalKFinder>();
builder.Services.AddScoped<IOptimalKFinderAggregator, OptimalKFinderAggregator>();
builder.Services.AddScoped<ISilhouetteScoreCalculator, SilhouetteScoreCalculator>();

builder.Services.AddScoped<IInertiaCalculator, InertiaCalculator>();
builder.Services.AddScoped<IDistanceCalculator, DistanceCalculator>();

builder.Services.AddScoped<IKMeansService, KMeansService>();
builder.Services.AddControllers();

var app = builder.Build();

app.UseAuthorization();
app.MapControllers();

app.Run();
