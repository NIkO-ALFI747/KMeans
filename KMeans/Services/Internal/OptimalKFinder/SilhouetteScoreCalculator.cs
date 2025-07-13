using KMeans.Services.Interfaces.Internal;
using KMeans.Services.Interfaces.Internal.IOptimalKFinder;

namespace KMeans.Services.Internal.OptimalKFinder;

public class SilhouetteScoreCalculator(IDistanceCalculator distanceCalculator) : ISilhouetteScoreCalculator
{
    private readonly IDistanceCalculator _distanceCalculator = distanceCalculator;

    public double CalculateAverageSilhouetteScore(float[][] data, int[] assignments)
    {
        int k = GetNumberOfClusters(assignments);
        if (!IsValidKForSilhouette(k, data.Length)) return 0;
        double totalSilhouetteScore = 0;
        for (int i = 0; i < data.Length; i++)
            totalSilhouetteScore += CalculateSilhouetteContributionForPoint(data, assignments, i, k);
        return totalSilhouetteScore / data.Length;
    }

    private double CalculateSilhouetteContributionForPoint(float[][] data, int[] assignments, int pointIndex, int k)
    {
        float[] currentPoint = data[pointIndex];
        int ownClusterId = assignments[pointIndex];
        double intraClusterDistance = CalculateIntraClusterDistance(currentPoint, ownClusterId, pointIndex, data, assignments);
        double minInterClusterDistance = CalculateMinInterClusterDistance(currentPoint, ownClusterId, k, data, assignments);
        return CalculateSingleSilhouetteScore(intraClusterDistance, minInterClusterDistance);
    }

    private static int GetNumberOfClusters(int[] assignments)
    {
        if (assignments.Length == 0) return 0;
        return assignments.Max() + 1;
    }

    private static bool IsValidKForSilhouette(int k, int dataLength)
    {
        return k > 1 && k < dataLength;
    }

    private double CalculateMinInterClusterDistance(float[] point, int ownClusterId, int k, float[][] data, int[] assignments)
    {
        double minAvgDistanceToNeighbor = double.MaxValue;
        for (int clusterId = 0; clusterId < k; clusterId++)
        {
            if (clusterId == ownClusterId) continue;
            double currentAvgDistance = CalculateAverageDistanceToSpecificCluster(point, clusterId, data, assignments);
            minAvgDistanceToNeighbor = Math.Min(minAvgDistanceToNeighbor, currentAvgDistance);
        }
        return minAvgDistanceToNeighbor;
    }

    private (double sumDistances, int count) SumDistancesToClusterPoints(float[] referencePoint, int targetClusterId,
        float[][] data, int[] assignments, int? excludeSelfIndex = null)
    {
        double sumDistances = 0;
        int count = 0;
        for (int j = 0; j < data.Length; j++)
        {
            if (assignments[j] == targetClusterId && (excludeSelfIndex == null || j != excludeSelfIndex.Value))
            {
                sumDistances += _distanceCalculator.GetDistance(referencePoint, data[j]);
                count++;
            }
        }
        return (sumDistances, count);
    }

    private double CalculateIntraClusterDistance(float[] point, int ownClusterId, int pointIndex, float[][] data, int[] assignments)
    {
        var (sumDistances, ownClusterSize) = SumDistancesToClusterPoints(point, ownClusterId, data, assignments, pointIndex);
        return ownClusterSize > 0 ? sumDistances / ownClusterSize : 0;
    }

    private double CalculateAverageDistanceToSpecificCluster(float[] point, int targetClusterId, float[][] data, int[] assignments)
    {
        var (totalDistToCluster, clusterSize) = SumDistancesToClusterPoints(point, targetClusterId, data, assignments);
        return clusterSize > 0 ? totalDistToCluster / clusterSize : double.MaxValue;
    }

    private static double CalculateSingleSilhouetteScore(double intraClusterDistance, double minInterClusterDistance)
    {
        var edgeCaseResult = HandleInfiniteMinInterClusterDistance(intraClusterDistance, minInterClusterDistance);
        if (edgeCaseResult.HasValue) return edgeCaseResult.Value;
        if (IsZeroDistanceEdgeCase(intraClusterDistance, minInterClusterDistance)) return 0;
        return CalculateStandardSilhouette(intraClusterDistance, minInterClusterDistance);
    }

    private static double? HandleInfiniteMinInterClusterDistance(double intraClusterDistance, double minInterClusterDistance)
    {
        if (double.IsInfinity(minInterClusterDistance) || minInterClusterDistance == double.MaxValue)
        {
            if (intraClusterDistance == 0 && minInterClusterDistance == double.MaxValue) return 0;
            if (intraClusterDistance > 0 && minInterClusterDistance == double.MaxValue) return 0;
            return 1.0;
        }
        return null;
    }

    private static bool IsZeroDistanceEdgeCase(double intraClusterDistance, double minInterClusterDistance)
    {
        return intraClusterDistance == 0 && minInterClusterDistance == 0;
    }

    private static double CalculateStandardSilhouette(double intraClusterDistance, double minInterClusterDistance)
    {
        if (Math.Max(intraClusterDistance, minInterClusterDistance) > 0)
            return (minInterClusterDistance - intraClusterDistance) / Math.Max(intraClusterDistance, minInterClusterDistance);
        return 0;
    }
}