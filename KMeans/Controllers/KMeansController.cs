using KMeans.Contracts;
using KMeans.Services.Interfaces;
using Microsoft.AspNetCore.Mvc;

namespace KMeans.Controllers;

[ApiController]
[Route("[controller]")]
public class ClusteringController(IKMeansService kmeans) : ControllerBase
{
    private readonly IKMeansService _kmeans = kmeans;

    [HttpPost("find-optimal-k")]
    public ActionResult<int> FindOptimalK([FromBody] OptimalKRequest request)
    {
        if (request?.Data == null || request.Data.Length == 0 || request.MaxK < 1)
            return BadRequest("Invalid request data. Provide data and a MaxK >= 1.");
        var optimalK = _kmeans.FindMostFrequentOptimalK(request.Data, request.MaxK);
        return Ok(optimalK);
    }
    
    [HttpPost("run-kmeans")]
    public ActionResult<KMeansResponse> RunKMeans([FromBody] KMeansRequest request)
    {
        if (request?.Data == null || request.Data.Length == 0 || request.K <= 0)
            return BadRequest("Invalid request data.");
        var (centroids, assignments, _) = _kmeans.RunKMeansMultiple(request.Data, request.K);
        return Ok(new KMeansResponse(centroids, assignments));
    }
}
