using KMeans.Services.Interfaces.Internal;
using KMeans.Services.Interfaces.Internal.IKMeansAlgorithm;

namespace KMeans.Services.Internal.KMeansAlgorithm;

public class KMeansAlgorithmAggregator(IKMeansAlgorithm kMeansAlgorithm, IInertiaCalculator inertiaCalculator) : IKMeansAlgorithmAggregator
{
    private readonly IKMeansAlgorithm _kMeansAlgorithm = kMeansAlgorithm;

    private readonly IInertiaCalculator _inertiaCalculator = inertiaCalculator;

    public (float[][] Centroids, int[] Assignments, double Inertia) RunKMeansMultiple(float[][] data, int k, int numRestarts = 5)
    {
        var (bestCentroids, bestAssignments, minTotalInertia) = InitializeBestKMeansResult();
        HashSet<int> usedInitialCentroidIndices = [];
        Random randomGenerator = new (Guid.NewGuid().GetHashCode());
        for (int run = 0; run < numRestarts; run++)
        {
            (bestCentroids, bestAssignments, minTotalInertia) = ProcessSingleKMeansRunIteration(data, k, bestCentroids, 
                bestAssignments, minTotalInertia, randomGenerator, usedInitialCentroidIndices);
        }
        return (bestCentroids, bestAssignments, minTotalInertia);
    }

    private static (float[][] bestCentroids, int[] bestAssignments, double minTotalInertia) InitializeBestKMeansResult()
    {
        return ([[]], [], double.MaxValue);
    }

    private (float[][] bestCentroids, int[] bestAssignments, double minTotalInertia) ProcessSingleKMeansRunIteration(float[][] data,
        int k, float[][] currentOverallBestCentroids, int[] currentOverallBestAssignments, double currentOverallMinTotalInertia,
        Random randomGenerator, HashSet<int> usedInitialCentroidIndices)
    {
        var (runCentroids, runAssignments, runInertia) = PerformSingleKMeansRun(data, k, randomGenerator, usedInitialCentroidIndices);
        return UpdateBestKMeansResult(runCentroids, runAssignments, runInertia, currentOverallBestCentroids, 
            currentOverallBestAssignments, currentOverallMinTotalInertia);
    }

    private (float[][] Centroids, int[] Assignments, double Inertia) PerformSingleKMeansRun(float[][] data, int k, 
        Random randomGenerator, HashSet<int> usedInitialCentroidIndices)
    {
        var (centroids, assignments) = _kMeansAlgorithm.RunKMeans(data, k, randomGenerator, usedInitialCentroidIndices);
        double inertia = _inertiaCalculator.CalculateInertia(data, centroids, assignments);
        return (centroids, assignments, inertia);
    }

    private static (float[][] bestCentroids, int[] bestAssignments, double minTotalInertia) UpdateBestKMeansResult(
        float[][] currentCentroids, int[] currentAssignments, double currentInertia,
        float[][] existingBestCentroids, int[] existingBestAssignments, double existingMinTotalInertia)
    {
        if (currentInertia < existingMinTotalInertia)
            return (currentCentroids, currentAssignments, currentInertia);
        else
            return (existingBestCentroids, existingBestAssignments, existingMinTotalInertia);
    }
}
