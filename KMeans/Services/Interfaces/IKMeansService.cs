namespace KMeans.Services.Interfaces;

public interface IKMeansService
{
    int FindMostFrequentOptimalK(float[][] data, int maxK, int numRuns = 10, double minAcceptableSilhouetteScore = 0.25);

    (float[][] Centroids, int[] Assignments, double Inertia) RunKMeansMultiple(float[][] data, int k, int numRestarts = 5);
}
