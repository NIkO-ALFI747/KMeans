using KMeans.Services.Interfaces.Internal.IOptimalKFinder;

namespace KMeans.Services.Internal.OptimalKFinder;

public class OptimalKFinderAggregator(IOptimalKFinder optimalKFinder) : IOptimalKFinderAggregator
{
    private readonly IOptimalKFinder _optimalKFinder = optimalKFinder;

    public int FindMostFrequentOptimalK(float[][] data, int maxK, int numRuns = 10, double minAcceptableSilhouetteScore = 0.25)
    {
        ValidateNumRuns(numRuns);
        var kCounts = new Dictionary<int, int>();
        PopulateKCountsFromMultipleRuns(data, maxK, numRuns, minAcceptableSilhouetteScore, kCounts);
        return GetMostFrequentK(kCounts);
    }

    private static void ValidateNumRuns(int numRuns)
    {
        if (numRuns <= 0) throw new ArgumentException("numRuns must be at least 1.");
    }

    private void PopulateKCountsFromMultipleRuns(float[][] data, int maxK, int numRuns,
        double minAcceptableSilhouetteScore, Dictionary<int, int> kCounts)
    {
        for (int i = 0; i < numRuns; i++)
        {
            var (finalKForRun, _) = _optimalKFinder.FindOptimalElbowKWithK1SilhouetteScore(data, maxK, minAcceptableSilhouetteScore);
            IncrementKCount(kCounts, finalKForRun);
        }
    }

    private static void IncrementKCount(Dictionary<int, int> kCounts, int kValue)
    {
        if (kCounts.TryGetValue(kValue, out int kCount)) kCounts[kValue] = ++kCount;
        else kCounts[kValue] = 1;
    }

    private static int GetMostFrequentK(Dictionary<int, int> kCounts)
    {
        if (kCounts.Count == 0) return 1;
        return kCounts.OrderByDescending(kv => kv.Value).ThenBy(kv => kv.Key).First().Key;
    }
}
