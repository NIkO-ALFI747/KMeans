namespace KMeans.Services.Interfaces.Internal.IOptimalKFinder;

public interface IOptimalKFinder
{
    (int FinalK, double SilhouetteScore) FindOptimalElbowKWithK1SilhouetteScore(float[][] data, int maxK, double minAcceptableSilhouetteScore = 0.25);
}
