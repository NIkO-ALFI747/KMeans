namespace KMeans.Services.Interfaces.Internal;

public interface IDistanceCalculator
{
    double GetDistance(float[] point1, float[] point2);

    double GetDistanceSquared(float[] point1, float[] point2);
}
