using KMeans.Services.Interfaces.Internal;

namespace KMeans.Services.Internal;

public class DistanceCalculator : IDistanceCalculator
{
    public double GetDistanceSquared(float[] point1, float[] point2)
    {
        return (double)point1.Zip(point2, (a, b) => (a - b) * (a - b)).Sum();
    }

    public double GetDistance(float[] point1, float[] point2)
    {
        return Math.Sqrt(GetDistanceSquared(point1, point2));
    }
}
