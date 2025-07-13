namespace KMeans.Services.Interfaces.Internal.IOptimalKFinder;

public interface IElbowMethodCalculator
{
    int FindOptimalElbowK(float[][] data, int maxK, double minPossibleDifference);
}
