using KMeans.Services.Interfaces.Internal;
using KMeans.Services.Interfaces.Internal.IKMeansAlgorithm;

namespace KMeans.Services.Internal.KMeansAlgorithm;

public class KMeansInitializer(IDistanceCalculator distanceCalculator) : IKMeansInitializer
{
    private readonly IDistanceCalculator _distanceCalculator = distanceCalculator;

    public float[][] InitializeCentroids(float[][] data, int k, Random randomGenerator, HashSet<int>? usedInitialCentroidIndices)
    {
        return InitializeCentroidsPlusPlus(data, k, randomGenerator, usedInitialCentroidIndices);
    }

    private float[][] InitializeCentroidsPlusPlus(float[][] data, int k, Random randomGenerator, HashSet<int>? usedInitialCentroidIndices)
    {
        var centroids = new List<float[]>{ GetRandomDataPoint(data, randomGenerator, usedInitialCentroidIndices) };
        for (int i = 1; i < k; i++)
        {
            double[] minDistances = CalculateMinDistancesToCentroids(data, centroids);
            int nextCentroidIndex = FindDataPointFurthestFromCentroids(minDistances);
            centroids.Add(data[nextCentroidIndex]);
        }
        return [.. centroids];
    }

    private static float[] GetRandomDataPoint(float[][] data, Random randomGenerator, HashSet<int>? usedInitialCentroidIndices)
    {
        int totalDataPoints = data.Length;
        if (usedInitialCentroidIndices == null) return GetRandomPointWithoutUniquenessCheck(data, randomGenerator, totalDataPoints);
        if (usedInitialCentroidIndices.Count == totalDataPoints) return GetRandomExistingPointWhenAllUsed(data, randomGenerator, usedInitialCentroidIndices, totalDataPoints);
        int actualDataIndex = FindUniqueAvailableIndex(randomGenerator, usedInitialCentroidIndices, totalDataPoints);
        ValidateAndMarkIndexAsUsed(actualDataIndex, usedInitialCentroidIndices);
        return data[actualDataIndex];
    }

    private static float[] GetRandomPointWithoutUniquenessCheck(float[][] data, Random randomGenerator, int totalDataPoints)
    {
        return data[randomGenerator.Next(totalDataPoints)];
    }

    private static float[] GetRandomExistingPointWhenAllUsed(float[][] data, Random randomGenerator, HashSet<int> usedInitialCentroidIndices, int totalDataPoints)
    {
        Console.WriteLine($"Warning: All {totalDataPoints} data points have been used as initial centroids for restarts. Reusing a random existing point.");
        return data[usedInitialCentroidIndices.ElementAt(randomGenerator.Next(usedInitialCentroidIndices.Count))];
    }

    private static int FindUniqueAvailableIndex(Random randomGenerator, HashSet<int> usedInitialCentroidIndices, int totalDataPoints)
    {
        int numAvailableIndexes = totalDataPoints - usedInitialCentroidIndices.Count;
        int targetAvailableSlot = randomGenerator.Next(numAvailableIndexes);
        return SearchForTargetAvailableIndex(usedInitialCentroidIndices, totalDataPoints, targetAvailableSlot);
    }

    private static int SearchForTargetAvailableIndex(HashSet<int> usedInitialCentroidIndices, int totalDataPoints, int targetAvailableSlot)
    {
        int actualDataIndex = -1;
        int currentAvailableCount = 0;
        for (int i = 0; i < totalDataPoints; i++)
        {
            (currentAvailableCount, actualDataIndex, bool targetFoundAndBreak) = ProcessIndexForTargetSlot(i, usedInitialCentroidIndices, targetAvailableSlot, currentAvailableCount);
            if (targetFoundAndBreak) break;
        }
        return actualDataIndex;
    }

    private static (int newCurrentAvailableCount, int foundActualDataIndex, bool targetFoundAndBreak) ProcessIndexForTargetSlot(int currentIndex, 
        HashSet<int> usedInitialCentroidIndices, int targetAvailableSlot, int currentAvailableCount)
    {
        int foundActualDataIndex = -1;
        bool targetFoundAndBreak = false;
        if (!usedInitialCentroidIndices.Contains(currentIndex))
        {
            if (currentAvailableCount == targetAvailableSlot)
            {
                foundActualDataIndex = currentIndex;
                targetFoundAndBreak = true;
            }
            currentAvailableCount++;
        }
        return (currentAvailableCount, foundActualDataIndex, targetFoundAndBreak);
    }

    private static void ValidateAndMarkIndexAsUsed(int actualDataIndex, HashSet<int> usedInitialCentroidIndices)
    {
        if (actualDataIndex == -1)
            throw new InvalidOperationException("Failed to find a unique initial centroid index despite available slots.");
        usedInitialCentroidIndices.Add(actualDataIndex);
    }

    private double[] CalculateMinDistancesToCentroids(float[][] data, List<float[]> centroids)
    {
        var minDistances = new double[data.Length];
        for (int j = 0; j < data.Length; j++)
            minDistances[j] = centroids.Min(c => _distanceCalculator.GetDistance(data[j], c));
        return minDistances;
    }

    private static int FindDataPointFurthestFromCentroids(double[] minDistances)
    {
        int nextCentroidIndex = 0;
        double maxDist = 0;
        for (int j = 0; j < minDistances.Length; j++)
        {
            if (minDistances[j] > maxDist)
            {
                maxDist = minDistances[j];
                nextCentroidIndex = j;
            }
        }
        return nextCentroidIndex;
    }
}