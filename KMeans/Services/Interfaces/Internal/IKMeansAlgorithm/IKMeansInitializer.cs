namespace KMeans.Services.Interfaces.Internal.IKMeansAlgorithm;

public interface IKMeansInitializer
{
    float[][] InitializeCentroids(float[][] data, int k, Random randomGenerator, HashSet<int>? usedInitialCentroidIndices);
}
