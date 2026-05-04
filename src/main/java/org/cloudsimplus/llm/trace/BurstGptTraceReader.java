package org.cloudsimplus.llm.trace;

import org.cloudsimplus.llm.core.LlmCloudlet;
import org.cloudsimplus.llm.workload.LlmModelSpec;

import java.io.BufferedReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Reader for the BurstGPT trace (Wang et al., 2024). Format:
 * <pre>request_id,timestamp_sec,prompt_tokens,response_tokens,model,success</pre>
 *
 * <p>Used in §6.6 (bursty workload autoscaling) because BurstGPT exhibits
 * heavy-tailed inter-arrival times that stress warm-pool sizing.</p>
 */
public class BurstGptTraceReader {

    private final LlmModelSpec defaultModel;

    public BurstGptTraceReader(LlmModelSpec defaultModel) {
        this.defaultModel = defaultModel;
    }

    public List<AzureLlmTraceReader.TraceEntry> read(Path csv) throws Exception {
        List<AzureLlmTraceReader.TraceEntry> out = new ArrayList<>();
        try (BufferedReader br = Files.newBufferedReader(csv)) {
            String header = br.readLine();
            if (header == null) return out;
            String line;
            long idCounter = 0;
            while ((line = br.readLine()) != null) {
                if (line.isBlank() || line.startsWith("#")) continue;
                String[] f = line.split(",");
                if (f.length < 4) continue;
                double tsSec = Double.parseDouble(f[1].trim());
                int promptTokens = Integer.parseInt(f[2].trim());
                int respTokens   = Integer.parseInt(f[3].trim());
                boolean success = f.length < 6 || Boolean.parseBoolean(f[5].trim());
                if (!success || promptTokens <= 0 || respTokens <= 0) continue;
                LlmCloudlet c = new LlmCloudlet(idCounter++, defaultModel,
                    promptTokens, respTokens, LlmCloudlet.SloClass.INTERACTIVE);
                out.add(new AzureLlmTraceReader.TraceEntry(tsSec, c));
            }
        }
        return out;
    }
}
