using KMeans.Services.Interfaces.Internal.IKMeansAlgorithm;
using KMeans.Services.Interfaces.Internal.IOptimalKFinder;

namespace KMeans.Services.Internal.OptimalKFinder;

public class OptimalKFinder(IKMeansAlgorithmAggregator kMeansAlgorithmAggregator, 
    IElbowMethodCalculator elbowMethodCalculator, ISilhouetteScoreCalculator silhouetteScoreCalculator) : IOptimalKFinder
{
    private readonly IKMeansAlgorithmAggregator _kMeansAlgorithmAggregator = kMeansAlgorithmAggregator;

    private readonly IElbowMethodCalculator _elbowMethodCalculator = elbowMethodCalculator;

    private readonly ISilhouetteScoreCalculator _silhouetteScoreCalculator = silhouetteScoreCalculator;

    private const double DefaultMinPossibleDifference = 10.0;

    private const int ElbowKMaxThreshold = 7;

    private const double InitialMinPossibleDifferenceForSmallK = 30.0;

    public (int FinalK, double SilhouetteScore) FindOptimalElbowKWithK1SilhouetteScore(float[][] data, int maxK, double minAcceptableSilhouetteScore = 0.25)
    {
        double minPossibleDifference = GetElbowMinPossibleDifference(maxK);
        int elbowK = CalculateElbowK(data, maxK, minPossibleDifference);
        var (silhouetteScore, _) = RunKMeansAndGetSilhouette(data, elbowK);
        int finalK = DetermineFinalKBasedOnSilhouette(elbowK, silhouetteScore, minAcceptableSilhouetteScore);
        return (finalK, silhouetteScore);
    }

    private static double GetElbowMinPossibleDifference(int maxK)
    {
        if (maxK > ElbowKMaxThreshold) return DefaultMinPossibleDifference;
        return InitialMinPossibleDifferenceForSmallK;
    }

    private int CalculateElbowK(float[][] data, int maxK, double minPossibleDifference)
    {
        int targetMaxKForInertia = maxK > 1 ? maxK + 1 : maxK;
        return _elbowMethodCalculator.FindOptimalElbowK(data, targetMaxKForInertia, minPossibleDifference);
    }

    private (double SilhouetteScore, int[] Assignments) RunKMeansAndGetSilhouette(float[][] data, int k)
    {
        var (_, assignments, _) = _kMeansAlgorithmAggregator.RunKMeansMultiple(data, k);
        double silhouetteScore = _silhouetteScoreCalculator.CalculateAverageSilhouetteScore(data, assignments);
        return (silhouetteScore, assignments);
    }

    private static int DetermineFinalKBasedOnSilhouette(int elbowK, double silhouetteScore, double minAcceptableSilhouetteScore)
    {
        if (silhouetteScore > minAcceptableSilhouetteScore) return elbowK;
        return 1;
    }
}
