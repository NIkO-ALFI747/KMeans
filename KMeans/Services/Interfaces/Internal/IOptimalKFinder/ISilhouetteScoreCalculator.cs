namespace KMeans.Services.Interfaces.Internal.IOptimalKFinder;

public interface ISilhouetteScoreCalculator
{
    double CalculateAverageSilhouetteScore(float[][] data, int[] assignments);
}
