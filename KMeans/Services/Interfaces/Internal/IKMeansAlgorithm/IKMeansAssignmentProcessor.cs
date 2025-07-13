namespace KMeans.Services.Interfaces.Internal.IKMeansAlgorithm;

public interface IKMeansAssignmentProcessor
{
    bool AssignPointsToClusters(float[][] data, float[][] centroids, int[] assignments);
}
