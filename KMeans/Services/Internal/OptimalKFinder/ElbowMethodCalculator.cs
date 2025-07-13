using KMeans.Services.Interfaces.Internal.IKMeansAlgorithm;
using KMeans.Services.Interfaces.Internal.IOptimalKFinder;

namespace KMeans.Services.Internal.OptimalKFinder;

public class ElbowMethodCalculator(IKMeansAlgorithmAggregator kMeansAlgorithmAggregator) : IElbowMethodCalculator
{
    private readonly IKMeansAlgorithmAggregator _kMeansAlgorithmAggregator = kMeansAlgorithmAggregator;

    private const double DefaultMinPossibleDifference = 10.0;

    private const double DefaultRatioTolerance = 0.1;

    public int FindOptimalElbowK(float[][] data, int maxK, double minPossibleDifference = DefaultMinPossibleDifference)
    {
        ValidateMaxK(maxK);
        if (IsSingleKCase(maxK)) return 1;
        var inertias = GenerateInertiaValuesForKRange(data, maxK);
        return FindOptimalElbowK(inertias, minPossibleDifference);
    }

    private static void ValidateMaxK(int maxK)
    {
        if (maxK < 1) throw new ArgumentException("maxK must be at least 1.");
    }

    private static bool IsSingleKCase(int maxK)
    {
        return maxK == 1;
    }

    private List<double> GenerateInertiaValuesForKRange(float[][] data, int maxK)
    {
        var inertias = new List<double>();
        for (int k = 1; k <= maxK; k++)
        {
            var (_, _, inertia) = _kMeansAlgorithmAggregator.RunKMeansMultiple(data, k);
            inertias.Add(inertia);
        }
        return inertias;
    }

    private static int FindOptimalElbowK(List<double> inertias, double minPossibleDifference = DefaultMinPossibleDifference,
        double ratioTolerance = DefaultRatioTolerance)
    {
        if (!IsValidInertiaListForElbow(inertias)) return 1;
        List<double> diffs = CalculateInertiaDifferences(inertias);
        List<double> diffRatios = CalculateDifferenceRatios(diffs, minPossibleDifference);
        int minIndex = FindElbowIndex(diffRatios, ratioTolerance);
        return minIndex + 2;
    }

    private static bool IsValidInertiaListForElbow(List<double> inertias)
    {
        return inertias.Count >= 2;
    }

    private static List<double> CalculateInertiaDifferences(List<double> inertias)
    {
        var diffs = new List<double>();
        for (int i = 0; i < inertias.Count - 1; i++)
            diffs.Add(inertias[i] - inertias[i + 1]);
        return diffs;
    }

    private static List<double> CalculateDifferenceRatios(List<double> diffs, double minPossibleDifference)
    {
        var diffRatios = new List<double>();
        for (int i = 0; i < diffs.Count - 1; i++)
        {
            double currentDiff = diffs[i];
            double nextDiff = diffs[i + 1];
            diffRatios.Add(DetermineRatioForDiffPair(currentDiff, nextDiff, minPossibleDifference));
        }
        return diffRatios;
    }

    private static double DetermineRatioForDiffPair(double currentDiff, double nextDiff, double minPossibleDifference)
    {
        if (currentDiff == 0) return double.PositiveInfinity;
        else if (nextDiff < 0 || currentDiff < 0) return double.MaxValue;
        else if (nextDiff < minPossibleDifference || currentDiff < minPossibleDifference) return double.MaxValue;
        else return nextDiff / currentDiff;
    }

    private static int FindElbowIndex(List<double> diffRatios, double ratioTolerance)
    {
        if (diffRatios.Count == 0) return 0;
        int minIndex = 0;
        double minRatio = double.MaxValue;
        for (int i = 0; i < diffRatios.Count; i++)
            (minRatio, minIndex) = ProcessRatioForElbow(diffRatios[i], i, minRatio, minIndex, ratioTolerance);
        return minIndex;
    }

    private static (double updatedMinRatio, int updatedMinIndex) ProcessRatioForElbow(
        double currentRatio, int currentIndex, double minRatio, int minIndex, double ratioTolerance)
    {
        if (currentRatio < minRatio)
        {
            minRatio = currentRatio;
            minIndex = currentIndex;
        }
        if (Math.Abs(currentRatio - minRatio) < ratioTolerance) minIndex = currentIndex;
        return (minRatio, minIndex);
    }
}
