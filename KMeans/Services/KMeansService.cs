using KMeans.Services.Interfaces;
using KMeans.Services.Interfaces.Internal.IKMeansAlgorithm;
using KMeans.Services.Interfaces.Internal.IOptimalKFinder;

namespace KMeans.Services;

public class KMeansService(IOptimalKFinderAggregator optimalKFinderAggregator,
    IKMeansAlgorithmAggregator kMeansAlgorithmAggregator) : IKMeansService
{
    private readonly IOptimalKFinderAggregator _optimalKFinderAggregator = optimalKFinderAggregator;

    private readonly IKMeansAlgorithmAggregator _kMeansAlgorithmAggregator = kMeansAlgorithmAggregator;

    public int FindMostFrequentOptimalK(float[][] data, int maxK, int numRuns = 10, double minAcceptableSilhouetteScore = 0.25)
    {
        return _optimalKFinderAggregator.FindMostFrequentOptimalK(data, maxK, numRuns, minAcceptableSilhouetteScore);
    }

    public (float[][] Centroids, int[] Assignments, double Inertia) RunKMeansMultiple(float[][] data, int k, int numRestarts = 5)
    {
        return _kMeansAlgorithmAggregator.RunKMeansMultiple(data, k, numRestarts);
    }
}
