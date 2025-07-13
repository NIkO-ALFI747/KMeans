using KMeans.Services.Interfaces.Internal.IKMeansAlgorithm;

namespace KMeans.Services.Internal.KMeansAlgorithm;

public class KMeansAlgorithm(IKMeansInitializer kmeansInitializer, IKMeansAssignmentProcessor assignmentProcessor, 
    IKMeansCentroidUpdater centroidUpdater) : IKMeansAlgorithm
{
    private readonly IKMeansInitializer _kmeansInitializer = kmeansInitializer;

    private readonly IKMeansAssignmentProcessor _assignmentProcessor = assignmentProcessor;
    
    private readonly IKMeansCentroidUpdater _centroidUpdater = centroidUpdater;

    public (float[][] Centroids, int[] Assignments) RunKMeans(float[][] data, int k, Random randomGenerator,
    HashSet<int>? usedInitialCentroidIndices = null, int maxIterations = 300)
    {
        float[][] centroids = _kmeansInitializer.InitializeCentroids(data, k, randomGenerator, usedInitialCentroidIndices);
        int[] assignments = InitializeAssignments(data.Length);
        PerformKMeansIterations(data, centroids, assignments, k, maxIterations);
        return (centroids, assignments);
    }

    private static int[] InitializeAssignments(int dataLength)
    {
        return new int[dataLength];
    }

    private void PerformKMeansIterations(float[][] data, float[][] centroids, int[] assignments, int k, int maxIterations)
    {
        bool changed = true;
        int iteration = 0;
        while (changed && iteration < maxIterations)
        {
            changed = _assignmentProcessor.AssignPointsToClusters(data, centroids, assignments);
            _centroidUpdater.UpdateCentroids(data, assignments, centroids, k);
            iteration++;
        }
    }
}
