"use client";

import { useState } from "react";
import { PageHeader } from "@/components/app-shell";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input, Select } from "@/components/ui/select";
import { api } from "@/lib/api";
import { usePolling } from "@/lib/use-polling";

const ROLES: [string, string][] = [["ANALYSTE", "Analyste"], ["SUPERVISEUR", "Superviseur"], ["ADMIN", "Administrateur"]];

export default function UsersPage() {
  const { data, reload } = usePolling<any[]>("/users", 0);
  const [form, setForm] = useState({ email: "", full_name: "", role: "ANALYSTE", password: "" });
  const [msg, setMsg] = useState<string | null>(null);

  async function create(e: React.FormEvent) {
    e.preventDefault(); setMsg(null);
    try { await api("/users", { method: "POST", body: JSON.stringify(form) }); setForm({ ...form, email: "", full_name: "", password: "" }); reload(); setMsg("Compte créé."); }
    catch (err: any) { setMsg(err.message); }
  }
  async function toggle(u: any) {
    await api(`/users/${u.id}`, { method: "PATCH", body: JSON.stringify({ is_active: !u.is_active }) }); reload();
  }

  return (
    <>
      <PageHeader title="Utilisateurs" subtitle="Comptes du back-office et rôles (RBAC)" />
      <div className="grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardContent className="overflow-x-auto p-0">
            <table className="w-full text-sm">
              <thead className="border-b bg-muted/50 text-left text-xs text-muted-foreground">
                <tr><th className="px-4 py-2.5">Nom</th><th>E-mail</th><th>Rôle</th><th>État</th><th className="px-4" /></tr>
              </thead>
              <tbody>
                {(data ?? []).map((u) => (
                  <tr key={u.id} className="border-b last:border-0">
                    <td className="px-4 py-2 font-medium">{u.full_name}</td><td className="text-xs">{u.email}</td>
                    <td className="text-xs">{ROLES.find(([r]) => r === u.role)?.[1]}</td>
                    <td className="text-xs">{u.is_active ? "Actif" : "Désactivé"}</td>
                    <td className="px-4 text-right"><Button size="sm" variant="outline" onClick={() => toggle(u)}>{u.is_active ? "Désactiver" : "Réactiver"}</Button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle>Nouveau compte</CardTitle></CardHeader>
          <CardContent>
            <form onSubmit={create} className="space-y-3">
              <Input placeholder="Nom complet" required value={form.full_name} onChange={(e) => setForm({ ...form, full_name: e.target.value })} />
              <Input placeholder="E-mail" type="email" required value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} />
              <Select className="w-full" value={form.role} onChange={(e) => setForm({ ...form, role: e.target.value })} options={ROLES} />
              <Input placeholder="Mot de passe (10 caractères min.)" type="password" minLength={10} required value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} />
              <Button type="submit" className="w-full">Créer le compte</Button>
              {msg && <p className="text-xs text-muted-foreground">{msg}</p>}
            </form>
          </CardContent>
        </Card>
      </div>
    </>
  );
}
