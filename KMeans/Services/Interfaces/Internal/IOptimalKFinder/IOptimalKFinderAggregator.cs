namespace KMeans.Services.Interfaces.Internal.IOptimalKFinder;

public interface IOptimalKFinderAggregator
{
    int FindMostFrequentOptimalK(float[][] data, int maxK, int numRuns = 10, double minAcceptableSilhouetteScore = 0.25);
}
