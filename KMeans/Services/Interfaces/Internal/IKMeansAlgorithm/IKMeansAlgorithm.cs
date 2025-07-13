namespace KMeans.Services.Interfaces.Internal.IKMeansAlgorithm;

public interface IKMeansAlgorithm
{
    (float[][] Centroids, int[] Assignments) RunKMeans(float[][] data, int k, Random randomGenerator, 
        HashSet<int>? usedInitialCentroidIndices = null, int maxIterations = 300);
}
