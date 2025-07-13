using KMeans.Services.Interfaces.Internal.IKMeansAlgorithm;

namespace KMeans.Services.Internal.KMeansAlgorithm;

public class KMeansCentroidUpdater() : IKMeansCentroidUpdater
{
    public void UpdateCentroids(float[][] data, int[] assignments, float[][] centroids, int k)
    {
        int numDimensions = centroids[0].Length;
        for (int i = 0; i < k; i++)
        {
            var clusterPoints = GetClusterPoints(data, assignments, i);
            if (clusterPoints.Count != 0) centroids[i] = CalculateClusterCentroid(clusterPoints, numDimensions);
        }
    }

    private static List<float[]> GetClusterPoints(float[][] data, int[] assignments, int clusterId)
    {
        return [.. data.Where((p, j) => assignments[j] == clusterId)];
    }

    private static float[] CalculateClusterCentroid(List<float[]> clusterPoints, int numDimensions)
    {
        return [.. clusterPoints.Aggregate(new float[numDimensions], (sum, p) =>
            [.. sum.Zip(p, (a, b) => a + b)])
            .Select(s => s / clusterPoints.Count)];
    }
}
