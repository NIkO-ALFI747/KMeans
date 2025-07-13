using KMeans.Services.Interfaces.Internal;
using KMeans.Services.Interfaces.Internal.IKMeansAlgorithm;

namespace KMeans.Services.Internal.KMeansAlgorithm;

public class KMeansAssignmentProcessor(IDistanceCalculator distanceCalculator) : IKMeansAssignmentProcessor
{
    private readonly IDistanceCalculator _distanceCalculator = distanceCalculator;

    public bool AssignPointsToClusters(float[][] data, float[][] centroids, int[] assignments)
    {
        bool changed = false;
        for (int i = 0; i < data.Length; i++)
        {
            int closestCentroidIndex = FindClosestCentroidIndex(data[i], centroids);
            changed |= UpdatePointAssignment(assignments, i, closestCentroidIndex);
        }
        return changed;
    }

    private int FindClosestCentroidIndex(float[] point, float[][] centroids)
    {
        int closestIndex = -1;
        double minDistance = double.MaxValue;
        for (int i = 0; i < centroids.Length; i++)
            (closestIndex, minDistance) = GetUpdatedClosestCentroidInfo(point, centroids[i], i, closestIndex, minDistance);
        return closestIndex;
    }

    private (int updatedClosestIndex, double updatedMinDistance) GetUpdatedClosestCentroidInfo(float[] point,
        float[] currentCentroid, int currentIndex, int currentClosestIndex, double currentMinDistance)
    {
        double distance = _distanceCalculator.GetDistance(point, currentCentroid);
        if (distance < currentMinDistance)
            return (currentIndex, distance);
        else
            return (currentClosestIndex, currentMinDistance);
    }

    private static bool UpdatePointAssignment(int[] assignments, int pointIndex, int newCentroidIndex)
    {
        if (assignments[pointIndex] != newCentroidIndex)
        {
            assignments[pointIndex] = newCentroidIndex;
            return true;
        }
        return false;
    }
}
