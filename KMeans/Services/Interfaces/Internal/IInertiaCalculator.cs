namespace KMeans.Services.Interfaces.Internal;

public interface IInertiaCalculator
{
    double CalculateInertia(float[][] data, float[][] centroids, int[] assignments);
}
