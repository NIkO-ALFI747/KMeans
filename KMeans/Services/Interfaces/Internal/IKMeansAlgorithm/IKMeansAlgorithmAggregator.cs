namespace KMeans.Services.Interfaces.Internal.IKMeansAlgorithm;

public interface IKMeansAlgorithmAggregator
{
    (float[][] Centroids, int[] Assignments, double Inertia) RunKMeansMultiple
        (float[][] data, int k, int numRestarts = 5);
}
