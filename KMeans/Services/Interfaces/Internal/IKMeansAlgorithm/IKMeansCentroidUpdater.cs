namespace KMeans.Services.Interfaces.Internal.IKMeansAlgorithm;

public interface IKMeansCentroidUpdater
{
    void UpdateCentroids(float[][] data, int[] assignments, float[][] centroids, int k);
}
