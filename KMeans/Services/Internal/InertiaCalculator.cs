using KMeans.Services.Interfaces.Internal;

namespace KMeans.Services.Internal;

public class InertiaCalculator(IDistanceCalculator distanceCalculator) : IInertiaCalculator
{
    private readonly IDistanceCalculator _distanceCalculator = distanceCalculator;

    public double CalculateInertia(float[][] data, float[][] centroids, int[] assignments)
    {
        double inertia = 0;
        for (int i = 0; i < data.Length; i++)
            inertia += _distanceCalculator.GetDistanceSquared(data[i], centroids[assignments[i]]);
        return inertia;
    }
}
